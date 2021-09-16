from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger

import torch

from CompNet.utils.io_utils import write_img
from CompNet.utils.obb_utils import match_xydir, orthonormal_xydir
from CompNet.utils.vis_utils import (
    convert_box_to_ply,
    get_n_colors,
    mask_color_img,
    color_img_mask,
)
from CompNet.utils.format_tool import html_from_template
from CompNet.viewer.tool import get_ncanvas_html


def logger_obj_html_unary(curr_obj, out_file, loss_list, model_name='model'):
    """Build visulizer tool with prediction and gt visualizing materials"""
    ncanvas_template_file = "./CompNet/viewer/templates/ply_ncanvas.html"

    img_list = [["./img.jpg", "./mask_img.jpg"]]
    ply_list = ["./pred.ply"]
    loss_list = [0.0]

    img_list.append(["./img.jpg", "./mask_img.jpg"])
    ply_list.append("./gt.ply")
    loss_list.append(0.0)

    ply_name_list = [
        f"'{Path(item).stem}'" for item in ply_list
    ]
    logger.info(f"Save html to {out_file}")
    get_ncanvas_html(
        img_list,
        ply_list,
        ply_name_list,
        curr_obj,
        model_name,
        ncanvas_template_file,
        out_file,
        loss_list=loss_list,
    )


def match_gt_size(pred_xydir, gt_xydir, gt_size):
    """
    Find corresponding ground truth axis length for prediction axes.
    """
    pred_xydir = torch.tensor(pred_xydir, dtype=torch.float).unsqueeze(0)
    gt_xydir = torch.tensor(gt_xydir, dtype=torch.float).unsqueeze(0)
    gt_size = torch.tensor(gt_size).float().unsqueeze(0)

    nn_inds = match_xydir(pred_xydir, gt_xydir)  # (b, 3)
    pred_size = torch.gather(gt_size, 1, nn_inds)
    pred_size = pred_size.squeeze(0).cpu().numpy()

    return pred_size


def file_logger_empty(data_batch, preds, output_dir, prefix, obj_name=None, cfg=None):
    """Empty file logger"""
    pass


def file_logger_relation(
    data_batch, preds, output_dir, prefix, obj_name=None, cfg=None
):
    """Output images and masks. Table list the label, whether its touch or not."""
    if prefix == "test":
        out_obj_dir = Path(output_dir) / obj_name
        out_obj_dir.mkdir(exist_ok=True, parents=True)

        batch_id = 0
        tp_colors = get_n_colors(2)

        # input image and masks
        masks = data_batch["mask"]
        mask1 = masks[batch_id, 0, :, :].detach().cpu().numpy()  # part1
        mask2 = masks[batch_id, 1, :, :].detach().cpu().numpy()  # part2

        img = data_batch["img"]
        img = img[0, 0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()

        mask1_img = mask_color_img(img, mask1, tp_colors[0])
        mask2_img = mask_color_img(img, mask2, tp_colors[1])

        write_img(str(out_obj_dir / f"mask1.jpg"), mask1_img)
        write_img(str(out_obj_dir / f"mask2.jpg"), mask2_img)


def file_logger_part(data_batch, preds, output_dir, prefix, obj_name=None, cfg=None):
    if cfg.MODEL.CHOICE in ["RotNet"]:
        use_gt_size = True  # predict rotation
    elif cfg.MODEL.CHOICE in ["AxisLengthNet", "GroupAxisLengthNet"]:
        use_gt_size = False  # predict size
    else:
        raise NotImplementedError()

    gt_box = data_batch["gt_box"]  # batch_size, num_parts, 12
    batch_size, num_parts, _ = gt_box.shape
    gt_center, gt_size, gt_xydir = gt_box[:, :, :3], gt_box[:, :, 3:6], gt_box[:, :, 6:]

    # input images
    img = data_batch["img"][0]
    if len(img.shape) == 4:
        img = img[0]  # num_part, C, H, W

    mask = data_batch["mask"][0]
    mask = mask.unsqueeze(dim=1)

    np_img = img.cpu().numpy()
    np_mask = mask.squeeze(dim=1).cpu().numpy()
    np_mask_img, part_colors = color_img_mask(np_img, np_mask, dataformat="CHW")
    mask_img = torch.from_numpy(np_mask_img)

    # shape predictions
    if all([x not in preds.keys() for x in ["center", "size", "xydir"]]):
        raise ValueError(
            f"Check prediction keys: {preds.keys()}. Module predicts nothing."
        )
    else:
        print(f"Prediction contains {preds.keys()}")

    if "center" in preds:
        pred_center = preds["center"].view(batch_size, num_parts, -1)
    else:
        print(f"Prediction box use ground truth center")
        pred_center = gt_center

    if use_gt_size:
        print(f"Prediction box use ground truth size")
        if "xydir" in preds:
            # match axis
            pred_xydir = preds["xydir"]
            gt_xydir = gt_xydir.view(batch_size * num_parts, 6)

            nn_inds = match_xydir(pred_xydir, gt_xydir)  # (b, 3)
            gt_size = gt_size.view(batch_size * num_parts, 3)  # (b, 3)
            pred_size = torch.gather(gt_size, 1, nn_inds)

            # align shape
            pred_size = pred_size.view(batch_size, num_parts, -1)
            gt_xydir = gt_xydir.view(batch_size, num_parts, -1)
        else:
            pred_size = gt_size
    else:
        assert "size" in preds
        pred_size = preds["size"]
        pred_size = pred_size.view(batch_size, num_parts, -1)

    if "xydir" in preds:
        pred_xydir = preds["xydir"]  # norm_xydir
        pred_xydir = orthonormal_xydir(pred_xydir)
        pred_xydir = pred_xydir.view(batch_size, num_parts, -1)
    else:
        print(f"Prediction box use ground truth orientation")
        pred_xydir = gt_xydir

    pred_shape = torch.cat(
        [pred_center, pred_size, pred_xydir], dim=2
    )  # b, num_parts, 12
    pred_box_list = pred_shape[0].detach().cpu().numpy()  # num_parts, 12
    gt_box_list = data_batch["gt_box"][0].cpu().numpy()  # (num_parts, 12)

    # export visualization
    out_obj_dir = Path(output_dir) / f"{obj_name}"
    out_obj_dir.mkdir(exist_ok=True)

    write_img(str(out_obj_dir / "img.jpg"), img.permute(1, 2, 0).detach().cpu().numpy())
    write_img(
        str(out_obj_dir / "mask_img.jpg"),
        mask_img.permute(1, 2, 0).detach().cpu().numpy(),
    )

    convert_box_to_ply(gt_box_list, part_colors, str(out_obj_dir / f"gt.ply"))
    convert_box_to_ply(pred_box_list, part_colors, str(out_obj_dir / f"pred.ply"))

    logger_obj_html_unary(
        obj_name, out_file=str(out_obj_dir / f"{obj_name}.html"), loss_list=[0.0], model_name=cfg.MODEL.CHOICE
    )


def file_logger_binary_center(
    data_batch, preds, output_dir, prefix, obj_name=None, cfg=None
):
    if prefix == "test":
        out_obj_dir = Path(output_dir) / obj_name
        out_obj_dir.mkdir(exist_ok=True)
        print(f"Save to {out_obj_dir}")

        batch_id = 0
        tp_colors = get_n_colors(2)
        # input image and masks
        masks = data_batch["mask"]
        mask1 = masks[batch_id, 0, :, :].detach().cpu().numpy()  # part1
        mask2 = masks[batch_id, 1, :, :].detach().cpu().numpy()  # part2

        img = data_batch["img"]
        img = img[0, 0, :, :, :].permute(1, 2, 0).detach().cpu().numpy()

        mask1_img = mask_color_img(img, mask1, tp_colors[0])
        mask2_img = mask_color_img(img, mask2, tp_colors[1])

        write_img(str(out_obj_dir / f"img.jpg"), mask1_img)
        write_img(str(out_obj_dir / f"mask_img.jpg"), mask2_img)

        # box1
        box_list = data_batch["gt_box"][batch_id].cpu().numpy()  # (2, 12)
        convert_box_to_ply(box_list, tp_colors, str(out_obj_dir / f"gt.ply"))

        gt_box1, gt_box2 = box_list[0, :], box_list[1, :]
        box2_center = preds["center"][batch_id, :].cpu().numpy()  # (batch_size, 3)
        pred_box2 = np.concatenate([box2_center, gt_box2[3:]])
        pred_box_list = np.array([gt_box1, pred_box2])
        convert_box_to_ply(pred_box_list, tp_colors, str(out_obj_dir / f"pred.ply"))

        logger_obj_html_unary(
            obj_name, out_file=str(out_obj_dir / f"{obj_name}.html"), loss_list=[0.0], model_name=cfg.MODEL.CHOICE
        )


def logger_table_html2(result, out_file, template_file="", suffix=""):
    """Result table html
           obj_index(link) | thumbnail | Loss | metric
      0       2501_ry15       img        0.01   0.01
      1         ...           ...        ...    ...

    Notes:
        Pandas library is used to convert table data to html string.
    """
    obj_data_format = (
        """<a href="./{obj}/{obj}{suffix}.html" title="{obj}"> {obj} </a>"""
    )
    img_data_format = """<figure class="image is-32x32"> <img src="./{obj}/img.jpg" alt="img"> </figure>"""

    table = []
    for item in result:
        obj_name, loss, metric = item
        table.append(
            [
                obj_data_format.format(obj=obj_name, suffix=suffix),
                img_data_format.format(obj=obj_name),
                loss,
                metric,
            ]
        )

    df = pd.DataFrame(table, columns=["Obj", "Img", "Loss", "Metric"])
    tmp = df.to_html(
        escape=False, classes="table sortable is-striped is-hoverable", border=0
    )

    html_from_template(template_file, out_file=out_file, table_string=tmp)


def logger_table_relation_html(result, out_file, template_file="", suffix=""):
    """Result table html
           obj_index(link) | thumbnail | thumbnail | Loss | metric
      0       2501_ry15        mask1       mask2      0.01   0.01
      1         ...           ...        ...    ...

    Notes:
        Pandas library is used to convert table data to html string.
    """
    obj_data_format = (
        """<a href="./{obj}/{obj}{suffix}.html" title="{obj}"> {obj} </a>"""
    )
    mask1_data_format = """<figure class="image is-32x32"> <img src="./{obj}/mask1.jpg" alt="img"> </figure>"""
    mask2_data_format = """<figure class="image is-32x32"> <img src="./{obj}/mask2.jpg" alt="img"> </figure>"""

    table = []
    for item in result:
        obj_name, loss, metric = item
        table.append(
            [
                obj_data_format.format(obj=obj_name, suffix=suffix),
                mask1_data_format.format(obj=obj_name, suffix=suffix),
                mask2_data_format.format(obj=obj_name, suffix=suffix),
                loss,
                metric,
            ]
        )

    df = pd.DataFrame(table, columns=["Obj", "Mask1", "Mask2", "Pred", "Label"])
    tmp = df.to_html(
        escape=False, classes="table sortable is-striped is-hoverable", border=0
    )

    html_from_template(template_file, out_file=out_file, table_string=tmp)
