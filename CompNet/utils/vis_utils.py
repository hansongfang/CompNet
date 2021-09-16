"""Tools for visualization"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import os.path as osp
from pathlib import Path
from functools import reduce
from loguru import logger

import math
import seaborn as sns
from CompNet.utils.io_utils import write_img, resize_img, export_ply_v_vc_f, read_img


def get_n_colors(n, cmap_name='rainbow'):
    cmap = plt.get_cmap(cmap_name)
    colors = np.arange(n).astype(np.float) / n
    colors = cmap(colors)[:, :3]

    return colors


def get_box_verts_faces(p, orthornormalize=True):
    """
        Args:
            p: [size, center, xydir] (3, 3, 6)
                - rotation is represented by the xydir
    """
    center = p[0: 3]
    lengths = p[3: 6]
    dir_1 = p[6: 9]
    dir_2 = p[9:]

    dir_1 = dir_1 / (np.linalg.norm(dir_1) + 1e-6)
    dir_2 = dir_2 / (np.linalg.norm(dir_2) + 1e-6)
    # assert np.isclose(np.dot(dir_1, dir_2), 0)
    dir_3 = np.cross(dir_1, dir_2)
    dir_3 = dir_3 / (np.linalg.norm(dir_3)+ 1e-6)
    if orthornormalize:
        old_dir_2 = dir_2
        dir_2 = np.cross(dir_3, dir_1)   # orthonormalizing
        # if not np.allclose(dir_2, old_dir_2):
        #     logger.info(f'Original xdir, ydir is not orthonormal')

    d1 = 0.5 * lengths[0] * dir_1
    d2 = 0.5 * lengths[1] * dir_2
    d3 = 0.5 * lengths[2] * dir_3

    verts = np.zeros([8, 3])
    verts[0][:] = center - d1 - d2 - d3  # (-1, -1, -1)
    verts[1][:] = center - d1 + d2 - d3  # (-1,  1, -1)
    verts[2][:] = center + d1 - d2 - d3  # ( 1, -1, -1)
    verts[3][:] = center + d1 + d2 - d3  # ( 1,  1, -1)
    verts[4][:] = center - d1 - d2 + d3  # (-1, -1,  1)
    verts[5][:] = center - d1 + d2 + d3  # (-1,  1,  1)
    verts[6][:] = center + d1 - d2 + d3  # ( 1, -1,  1)
    verts[7][:] = center + d1 + d2 + d3  # ( 1,  1,  1)

    faces = [[4, 7, 5], [7, 4, 6],  # front
             [1, 3, 0], [0, 3, 2],  # back
             [3, 1, 5], [3, 5, 7],  # top
             [4, 0, 2], [4, 2, 6],  # bottom
             [5, 1, 4], [4, 1, 0],  # side1
             [3, 7, 6], [3, 6, 2],  # side2
             ]
    faces = np.array(faces)

    return verts, faces


def get_shape_verts_colors_faces(box_list, part_colors):
    """

    Args:
        box_list: np.array, (num_parts, 12)
        part_colors: np.array, (num_parts, 3)

    Notes:
        return color in range 0.0 - 1.0

    """
    verts = []
    colors = []
    faces = []
    offset = 0
    for (id, box) in enumerate(box_list):
        part_color = part_colors[id]
        box_verts, box_faces = get_box_verts_faces(box)
        box_colors = np.ones((8, 3), dtype=np.float) * part_color
        box_colors = box_colors
        box_faces = box_faces + offset
        verts.append(box_verts)
        colors.append(box_colors)
        faces.append(box_faces)
        offset += 8

    verts = np.vstack(verts)
    colors = np.vstack(colors)
    faces = np.vstack(faces)

    verts = verts[np.newaxis, :, :]
    colors = colors[np.newaxis, :, :]
    faces = faces[np.newaxis, :, :]

    return verts, colors, faces


def convert_box_to_ply(box_list, color_list, out_file):
    verts, colors, faces = get_shape_verts_colors_faces(box_list,
                                                        color_list)
#     print(f'Export ply file to {out_file}')
    export_ply_v_vc_f(out_file,
                      verts[0],
                      colors[0],
                      faces[0])


def convert_obb_list_to_ply(out_file, obb_list, color_list=None):
    if color_list is None:
        color_list = get_n_colors(len(obb_list))
    verts, colors, faces = get_shape_verts_colors_faces(obb_list, color_list)
    print(f'Save ply file to {out_file}')
    export_ply_v_vc_f(out_file,
                      verts[0],
                      colors[0],
                      faces[0])


def mask_color_img(img, mask, mask_color, alpha=0.6):
    """
    Args:
        img: np.array, (H, W, C)
        mask: np.array, (H, W)
        mask_color: np.array, (3,)
        alpha: float
    """
    mask_img = mask[:, :, np.newaxis] * mask_color
    return alpha * img + (1 - alpha) * mask_img


def color_img_mask(img, mask, cmap_name='rainbow', alpha=0.6, dataformat='HWC'):
    """
    Args:
        img: np.array
        mask: np.array
        cmap_name:
        alpha: float

    Returns:

    """
    num_part, height, width = mask.shape
    ori_img = np.copy(img)

    cmap = plt.get_cmap(cmap_name)
    colors = np.arange(num_part).astype(np.float) / num_part
    colors = cmap(colors)[:, :3]

    if dataformat == 'HWC':
        all_mask = np.zeros((height, width, 3), dtype=np.float)
        for id in range(num_part):
            part_mask = mask[id]
            all_mask += part_mask[:, :, np.newaxis] * colors[id]
        write_img('./tt.png', all_mask)
        # mask_img = alpha * ori_img + (1 - alpha) * all_mask
        mask_img = 0.4 * ori_img + 0.5 * all_mask
    elif dataformat == 'CHW':
        all_mask = np.zeros((3, height, width), dtype=np.float)
        for id in range(num_part):
            part_mask = mask[id]
            part_color = colors[id]
            all_mask += part_mask[np.newaxis, :, :] * part_color[:, np.newaxis, np.newaxis]
        mask_img = alpha * ori_img + (1 - alpha) * all_mask

    return mask_img, colors


def draw_box(ax, p, color, rot=None):
    """
        Args:
            p: [size, center, xydir] (3, 3, 6)
                - rotation is represented by the xydir
    """
    center = p[0: 3]
    lengths = p[3: 6]
    dir_1 = p[6: 9]
    dir_2 = p[9:]

    if rot is not None:
        center = (rot * center.reshape(-1, 1)).reshape(-1)
        dir_1 = (rot * dir_1.reshape(-1, 1)).reshape(-1)
        dir_2 = (rot * dir_2.reshape(-1, 1)).reshape(-1)

    dir_1 = dir_1/np.linalg.norm(dir_1)
    dir_2 = dir_2/np.linalg.norm(dir_2)
    dir_3 = np.cross(dir_1, dir_2)
    dir_3 = dir_3/np.linalg.norm(dir_3)

    d1 = 0.5 * lengths[0] * dir_1
    d2 = 0.5 * lengths[1] * dir_2
    d3 = 0.5 * lengths[2] * dir_3

    cornerpoints = np.zeros([8, 3])
    cornerpoints[0][:] = center - d1 - d2 - d3
    cornerpoints[1][:] = center - d1 + d2 - d3
    cornerpoints[2][:] = center + d1 - d2 - d3
    cornerpoints[3][:] = center + d1 + d2 - d3
    cornerpoints[4][:] = center - d1 - d2 + d3
    cornerpoints[5][:] = center - d1 + d2 + d3
    cornerpoints[6][:] = center + d1 - d2 + d3
    cornerpoints[7][:] = center + d1 + d2 + d3

    ax.plot([cornerpoints[0][0], cornerpoints[1][0]], [cornerpoints[0][1], cornerpoints[1][1]],
            [cornerpoints[0][2], cornerpoints[1][2]], c=color)
    ax.plot([cornerpoints[0][0], cornerpoints[2][0]], [cornerpoints[0][1], cornerpoints[2][1]],
            [cornerpoints[0][2], cornerpoints[2][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[3][0]], [cornerpoints[1][1], cornerpoints[3][1]],
            [cornerpoints[1][2], cornerpoints[3][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[3][0]], [cornerpoints[2][1], cornerpoints[3][1]],
            [cornerpoints[2][2], cornerpoints[3][2]], c=color)
    ax.plot([cornerpoints[4][0], cornerpoints[5][0]], [cornerpoints[4][1], cornerpoints[5][1]],
            [cornerpoints[4][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[4][0], cornerpoints[6][0]], [cornerpoints[4][1], cornerpoints[6][1]],
            [cornerpoints[4][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[5][0], cornerpoints[7][0]], [cornerpoints[5][1], cornerpoints[7][1]],
            [cornerpoints[5][2], cornerpoints[7][2]], c=color)
    ax.plot([cornerpoints[6][0], cornerpoints[7][0]], [cornerpoints[6][1], cornerpoints[7][1]],
            [cornerpoints[6][2], cornerpoints[7][2]], c=color)
    ax.plot([cornerpoints[0][0], cornerpoints[4][0]], [cornerpoints[0][1], cornerpoints[4][1]],
            [cornerpoints[0][2], cornerpoints[4][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[5][0]], [cornerpoints[1][1], cornerpoints[5][1]],
            [cornerpoints[1][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[6][0]], [cornerpoints[2][1], cornerpoints[6][1]],
            [cornerpoints[2][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[3][0], cornerpoints[7][0]], [cornerpoints[3][1], cornerpoints[7][1]],
            [cornerpoints[3][2], cornerpoints[7][2]], c=color)


def draw_geo(ax, p, color, rot=None):
    if rot is not None:
        p = (rot * p.transpose()).transpose()

    ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=[color], marker='.')


def drawShape(box_array, box_color_array=None, vis_geo=False, geo_array=None, extent=1.0, save_fig=False, out_file=None, title=None):
    """Draw shape with box [center, size, xydir]

    Args:
        box_array:

    Returns:

    """
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-extent, extent)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    ax.set_proj_type('persp')

    # transform coordinates so z is up (from y up)
    coord_rot = np.matrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]])

    for id, box in enumerate(box_array):
        box_color = [1, 0, 0]
        if box_color_array.any():
            box_color = box_color_array[id]
        draw_box(ax, box, color=box_color, rot=coord_rot)

    if vis_geo:
        for geo in geo_array:
            draw_geo(ax, geo, color=[0, 1, 0], rot=coord_rot)

    # plt.tight_layout()

    if title:
        plt.title(title)

    if save_fig:
        print(f'Save figure to {out_file}')
        plt.savefig(out_file)

    # plt.show()
    plt.close()


def draw_edges(edges, nodes, img, save_fig=False, out_file=None):
    sns.set_context('paper')
    # img - mask - edge_type
    num_edges = len(edges)
    nw = math.floor(math.sqrt(num_edges)) + 1

    fig, ax = plt.subplots(nw, nw, figsize=(16, 16))
    for nx in range(nw):
        for ny in range(nw):
            subax = ax[ny, nx]
            subax.axis('off')

    for id, edge in enumerate(edges):
        edge_type, part_a, part_b = edge
        nx = id % nw
        ny = int(id / nw)

        a_mask = nodes[part_a]['mask']
        if len(a_mask.shape) == 3:
            a_mask = a_mask[:, :, 0]
        b_mask = nodes[part_b]['mask']
        if len(b_mask.shape) == 3:
            b_mask = b_mask[:, :, 0]

        edge_mask = np.stack([a_mask, b_mask])
        edge_mask_img, _ = color_img_mask(img, edge_mask, dataformat='HWC')

        subax = ax[ny, nx]
        subax.axis('off')
        subax.imshow(edge_mask_img)
        subax.set_title(f'{edge_type}: {part_a}-{part_b}')

    fig.tight_layout()
    if save_fig:
        fig.savefig(out_file)

    plt.close()


def vis_obj_data(obj_data_dir, vis_obj_dir, save_box=False, vis_edge=False):
    os.makedirs(vis_obj_dir, exist_ok=True)

    # img
    img_fn = Path(obj_data_dir)/'img.png'
    img = read_img(str(img_fn))
    write_img(osp.join(vis_obj_dir, 'img.jpg'), img)

    obj_fn = Path(obj_data_dir)/f'{Path(obj_data_dir).stem}.npy'
    obj_data = np.load(obj_fn, allow_pickle=True).item()

    part_nodes = obj_data['nodes']
    for pid in part_nodes.keys():
        part_mask_fn = Path(obj_data_dir) / f'partmask_{pid}.png'
        part_mask = read_img(str(part_mask_fn), 0)
        part_nodes[pid]['mask'] = part_mask

    part_list = obj_data['vis_parts']
    parts_mask = []
    parts_box = []
    partid_to_id = {}
    for x, part_id in enumerate(part_list):
        partid_to_id[part_id] = x
        part_node = obj_data['nodes'][part_id]
        part_mask = part_node['mask']
        parts_mask.append(part_mask)

        part_box = part_node['box']
        parts_box.append(part_box)

    parts_mask = np.array(parts_mask)
    parts_box = np.array(parts_box)

    # masked_img
    mask_img, parts_colors = color_img_mask(img, parts_mask, dataformat='HWC')
    write_img(osp.join(vis_obj_dir, 'mask_img.png'), mask_img)

    # box with specify color
    drawShape(parts_box,
              box_color_array=parts_colors,
              save_fig=True,
              out_file=osp.join(vis_obj_dir, 'gt_box.png'))

    if vis_edge:
        # edges
        draw_edges(edges=obj_data['edges'],
                   nodes=obj_data['nodes'],
                   img=img,
                   save_fig=True,
                   out_file=osp.join(vis_obj_dir, 'edges.png'))

    if save_box:
        verts, colors, faces = get_shape_verts_colors_faces(parts_box,
                                                            parts_colors)
        export_ply_v_vc_f(osp.join(vis_obj_dir, 'box_shape.ply'),
                          verts[0],
                          colors[0],
                          faces[0])



