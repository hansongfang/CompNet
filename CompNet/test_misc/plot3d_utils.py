"""
Modified from the plt 3d render utility code from pt2pc and structurenet

Original author: Kaichun
https://github.com/daerduoCarey/pt2pc/blob/master/utils.py
https://github.com/daerduoCarey/structurenet/blob/master/code/vis_utils.py

"""

import os
import sys
# import torch
import numpy as np
import matplotlib.pylab as plt
# from mpl_toolkits.mplot3d import Axes3D
# import trimesh

# colors = [[0, 0.8, 0], [0.8, 0, 0], [0, 0.3, 0], \
#         [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], \
#         [0.3, 0.6, 0], [0.6, 0, 0.3], [0.3, 0, 0.6], \
#         [0.6, 0.3, 0], [0.3, 0, 0.6], [0.6, 0, 0.3], \
#         [0.8, 0.2, 0.5], [0.8, 0.2, 0.5], [0.2, 0.8, 0.5], \
#         [0.2, 0.5, 0.8], [0.5, 0.2, 0.8], [0.5, 0.8, 0.2], \
#         [0.3, 0.3, 0.7], [0.3, 0.7, 0.3], [0.7, 0.3, 0.3]]

# [0, 0.3, 0], [0.3, 0, 0], [0, 0, 0.3],\
colors = [[0, 0.8, 0], [0.8, 0, 0], [0, 0, 0.8], \
        [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], \
        [0.3, 0.6, 0], [0.6, 0, 0.3], [0, 0.3, 0.6], \
        [0.6, 0.3, 0], [0.3, 0, 0.6], [0, 0.6, 0.3], \
        [0.8, 0.2, 0.5], [0.8, 0.5, 0.2], [0.2, 0.8, 0.5], \
        [0.2, 0.5, 0.8], [0.5, 0.2, 0.8], [0.5, 0.8, 0.2], \
        [0.3, 0.3, 0.7], [0.3, 0.7, 0.3], [0.7, 0.3, 0.3]]

# def render_pc(out_fn, pc, figsize=(8, 8)):
#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_subplot(1, 1, 1, projection='3d')
#     ax.view_init(elev=20, azim=60)
#     x = pc[:, 0]
#     y = pc[:, 2]
#     z = pc[:, 1]
#     ax.scatter(x, y, z, marker='.')
#     miv = np.min([np.min(x), np.min(y), np.min(z)])  # Multiply with 0.7 to squeeze free-space.
#     mav = np.max([np.max(x), np.max(y), np.max(z)])
#     ax.set_xlim(miv, mav)
#     ax.set_ylim(miv, mav)
#     ax.set_zlim(miv, mav)
#     plt.tight_layout()
#     fig.savefig(out_fn, bbox_inches='tight')
#     plt.close(fig)

def convert_color_to_hexcode(rgb):
    r, g, b = rgb
    return '#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))

def render_part_pcs(pcs_list, title_list=None, out_fn=None, \
        subplotsize=(1, 1), figsize=(8, 8), azim=60, elev=20, 
        scale=0.3, dpi=100, unit_scale_and_centered=True):
    """
    Render the given list of shape, where each shape consists of a list of ragged parts pcd,
    into plots of size `subplotsize`.
    The shape center is assumed to be unit scaled and close to origin 
    for proper axis tick setting.
    """
    num_pcs = len(pcs_list)
    fig = plt.figure(figsize=figsize, dpi=dpi)
    for k in range(num_pcs):
        pcs = pcs_list[k]
        ax = fig.add_subplot(subplotsize[0], subplotsize[1], k+1, projection='3d')
        ax.view_init(elev=elev, azim=azim)
        xs = []; ys = []; zs = [];
        for i in range(len(pcs)):
            x = pcs[i][:, 0]
            y = pcs[i][:, 2]
            z = pcs[i][:, 1]
            xs.append(x)
            ys.append(y)
            zs.append(z)
            # if out_fn is None:
            #     ax.scatter(x, y, z, marker='.', s=scale, c=convert_color_to_hexcode(colors[i % len(colors)]))
            # else:
            #     ax.scatter(x, y, z, marker='.', c=convert_color_to_hexcode(colors[i % len(colors)]))
            if scale is not None:
                ax.scatter(x, y, z, marker='.', s=scale, c=convert_color_to_hexcode(colors[i % len(colors)]))
            else:
                ax.scatter(x, y, z, marker='.', c=convert_color_to_hexcode(colors[i % len(colors)]))
        xs = np.concatenate(xs, axis=0)
        ys = np.concatenate(ys, axis=0)
        zs = np.concatenate(zs, axis=0)
        miv = np.min([np.min(xs), np.min(ys), np.min(zs)]) * 0.9 # zoom in a little
        mav = np.max([np.max(xs), np.max(ys), np.max(zs)]) * 0.9
        ax.set_xlim(miv, mav)
        ax.set_ylim(miv, mav)
        ax.set_zlim(miv, mav)
        if title_list is not None:
            ax.set_title(title_list[k], pad=0)
        if unit_scale_and_centered:
            ax.set_xticks([-0.5,0,0.5])
            ax.set_yticks([-0.5,0,0.5])
            ax.set_zticks([-0.5,0,0.5])
    plt.tight_layout()
    if out_fn is not None:
        fig.savefig(out_fn, bbox_inches='tight')
        plt.show()
        # plt.close(fig)
    else:
        plt.show()


### code to draw box, pt, edge
def draw_box(ax, p, color, rot=None):
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
    cornerpoints = np.zeros([8, 3])

    d1 = 0.5*lengths[0]*dir_1
    d2 = 0.5*lengths[1]*dir_2
    d3 = 0.5*lengths[2]*dir_3

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

def draw_edge(ax, e, p_from, p_to, rot=None):
    if rot is not None:
        center1 = (rot * p_from.reshape(-1, 1)).reshape(-1)[0]
        center2 = (rot * p_to.reshape(-1, 1)).reshape(-1)[0]

    edge_type_colors = {
        'ADJ': (1, 0, 0),
        'ROT_SYM': (1, 1, 0),
        'TRANS_SYM': (1, 0, 1),
        'REF_SYM': (0, 0, 0)}

    edge_type_linewidth = {
        'ADJ': 8,
        'ROT_SYM': 6,
        'TRANS_SYM': 4,
        'REF_SYM': 2}

    ax.plot(
        [center1[0, 0], center2[0, 0]], [center1[0, 1], center2[0, 1]], [center1[0, 2], center2[0, 2]],
        c=edge_type_colors[e['type']],
        linestyle=':',
        linewidth=edge_type_linewidth[e['type']])