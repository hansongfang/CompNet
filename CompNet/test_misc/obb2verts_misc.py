"""
Sample pts for obb surface
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import trimesh

from common_3d.ops.farthest_point_sample.farthest_point_sample import farthest_point_sample_wrapper


def get_rot_mat_from_xydir(dir_1, dir_2):
    xdir = dir_1 / np.linalg.norm(dir_1)
    ydir = dir_2 / np.linalg.norm(dir_2)
    # assert np.isclose(np.dot(dir_1, dir_2), 0)
    zdir = np.cross(xdir, ydir)
    zdir = zdir / np.linalg.norm(zdir)
    ydir = np.cross(zdir, xdir)   # orthonormalizing

    rotmat = np.stack([xdir, ydir, zdir], axis=-1)

    return rotmat


def sample_obb_verts(obb, nb_pts=1024):
    obb_center, obb_size, obb_xydir = obb[:3], obb[3:6], obb[6:]
    rot_mat = get_rot_mat_from_xydir(obb_xydir[:3],
                                     obb_xydir[3:])
    trans_mat = np.eye(4)
    trans_mat[:3, :3] = rot_mat
    trimesh_obb = trimesh.creation.box(obb_size, trans_mat)
    cube_verts, _ = trimesh.sample.sample_surface_even(trimesh_obb, nb_pts)
    cube_verts += obb_center

    if cube_verts.shape[0] != nb_pts:
        tmp_pts = np.pad(cube_verts,
                         ((0, nb_pts - cube_verts.shape[0]), (0, 0)),
                         mode='symmetric')
        # draw_pc_list([cube_verts])
        # draw_pc_list([tmp_pts])
        assert tmp_pts.shape[0] == nb_pts
        cube_verts = tmp_pts

    return cube_verts


def sample_obb_list_verts(obb_list, nb_pts=1024):
    """ sample nb_pts points from each obb and then do FPS """
    shape_verts = []
    for obb in obb_list:
        obb_verts = sample_obb_verts(obb, nb_pts)
        shape_verts.append(obb_verts)
    shape_verts = np.concatenate(shape_verts, axis=0)
    shape_fps_pts = farthest_point_sample_wrapper(shape_verts, nb_pts=nb_pts)
    return shape_fps_pts



