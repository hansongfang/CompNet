import numpy as np
import torch
import torch.nn.functional as F

from common.nn.functional import pdist2


def get_box_pts_torch(center=None, size=None, rx=None, ry=None, use_gpu="True"):
    """Orthonormalize rx and ry before computing.

    Args:
        center: torch.tensor, (batch_size, 3)
        size: torch.tensor, (batch_size, 3)
        rx: torch.tensor, (batch_size, 3), x axis
        ry: torch.tensor, (batch_size, 3), y axis

    Returns:

    """
    # unit_size box
    const = np.array(
        [
            [
                [-1, -1, -1],
                [-1, 1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, -1, 1],
                [-1, 1, 1],
                [1, -1, 1],
                [1, 1, 1],
            ]
        ],
        dtype=np.float32,
    )
    const = const * 0.5  # unit size box
    const = torch.from_numpy(const)
    if use_gpu:
        const = const.cuda()
    const.requires_grad = True

    box = const
    if size is not None:
        box = const * (size.unsqueeze(1).contiguous())
    if rx is not None and ry is not None:
        x = F.normalize(rx, dim=1, p=2)  # (batch_size, 3)
        z = torch.cross(x, ry, dim=1)
        z = F.normalize(z, dim=1, p=2)
        y = torch.cross(z, x, dim=1)
        rot = torch.stack([x, y, z], dim=1)
        rot = rot.view(-1, 3, 3)
        box = torch.matmul(box, rot)
    if center is not None:
        box = box + center.unsqueeze(dim=1)

    return box


def get_box_pts_np(center=None,
                   size=None,
                   rx=None,
                   ry=None):
    """

            1 -------- 3
           /|         /|
          5 -------- 7 .
          | |        | |
          . 0 -------- 2
          |/         |/
          4 -------- 6

    Args:
        center: np.array, (3, )
        size: np.array, (3, )
        rx: np.array, (3, )
        ry: np.array, (3, )

    Returns:

    """
    # unit_size box
    const = np.array(
        [
            [-1, -1, -1],
            [-1, 1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, -1, 1],
            [-1, 1, 1],
            [1, -1, 1],
            [1, 1, 1],
        ],
        dtype=np.float32,
    )
    const = const * 0.5  # unit size box

    box = const
    if size is not None:
        box = const * size
    if rx is not None and ry is not None:
        x = rx / np.linalg.norm(rx)
        y = ry / np.linalg.norm(ry)
        rz = np.cross(rx, ry)
        z = rz / np.linalg.norm(rz)

        rot_t = np.stack([x, y, z], axis=0)
        box = np.matmul(box, rot_t)

    if center is not None:
        box = box + center

    return box


def get_xyzdir(x_raw, y_raw):
    """Orthonormalize xdir, ydir and get zdir.

    Args:
        xdir: torch.tensor, (b, 3)
        ydir: torch.tensor, (b, 3)

    Returns:
        zdir: torch.tensor, (b, 3)
    """

    xdir = F.normalize(x_raw, dim=1, p=2)  #

    proj_y = torch.sum(y_raw * xdir, dim=1, keepdim=True) * xdir  # projection vector
    ydir = y_raw - proj_y
    ydir = F.normalize(ydir, dim=1, p=2)

    zdir = torch.cross(xdir, ydir, dim=1)
    zdir = F.normalize(zdir, dim=1, p=2)

    return xdir, ydir, zdir


def match_xydir(pred_xydir, gt_xydir):
    """

    Args:
        pred_xydir: torch.tensor, (batch_size, 6)
        gt_xydir: torch.tensor, (batch_size, 6)

    """
    gt_xdir, gt_ydir, gt_zdir = get_xyzdir(gt_xydir[:, :3], gt_xydir[:, 3:])
    pred_xdir, pred_ydir, pred_zdir = get_xyzdir(pred_xydir[:, :3], pred_xydir[:, 3:])

    # add signed axis
    gt_dirs = torch.stack([gt_xdir, gt_ydir, gt_zdir, -gt_xdir, -gt_ydir, -gt_zdir], dim=-1)  # (batch_size, C, npts)
    pred_dirs = torch.stack([pred_xdir, pred_ydir, pred_zdir, -pred_xdir, -pred_ydir, -pred_zdir], dim=-1)  # (batch_size, C, npts)

    distance = pdist2(pred_dirs, gt_dirs)
    _, nn_inds = torch.topk(distance, 1, largest=False, sorted=False)  # (batch_size, npts, 1)

    # remove signed axis
    nn_inds = nn_inds[:, :3, :]
    nn_inds = nn_inds % 3
    nn_inds = nn_inds.squeeze(dim=-1)

    return nn_inds


def orthonormal_xydir(xy_raw):
    """Gram–Schmidt process

    Args:
        xy_raw: torch.tensor, (n, 6)

    Returns:

    """
    assert len(xy_raw.shape) == 2
    x_raw = xy_raw[:, :3]
    y_raw = xy_raw[:, 3:6]

    x = F.normalize(x_raw, dim=1, p=2)  #
    proj_y = torch.sum(y_raw * x, dim=1, keepdim=True) * x  # projection vector
    y = y_raw - proj_y
    y = F.normalize(y, dim=1, p=2)
    xy_dir = torch.cat([x, y], dim=1)

    return xy_dir


def get_rotmat_from_xydir_v2(xydir, orthonormal=False):
    """
    Args:
        xydir: torch.tensor, (batch_size, 6)

    Returns:
        rotmat: torch.tensor, (batch_size, 3, 3)

    """
    if orthonormal:
        xydir = orthonormal_xydir(xydir)
    xdir, ydir = xydir[:, :3], xydir[:, 3:]
    zdir = torch.cross(xdir, ydir, dim=1)
    rot = torch.stack([xdir, ydir, zdir], dim=-1)

    return rot
