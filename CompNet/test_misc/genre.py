"""
GenRe
  - input: image
  - output: mesh.obj

Mesh.obj space (I don't know why GenRe don't use common opencv camera space):
   - center at [0, 0, 0]
   - x axis is opencv camera z axis
   - y aixs is opencv camera -y axis
   - z axis is opencv camera x axis

Mesh.obj
    - # vertices
    - # faces

Mesh normalization
    - GenRe use the shapeNet data.
    - ShapeNet data is normalized to ?
    - PartNet data is normalized to ?

Songfang Han
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import open3d as o3d
import numpy as np
np.set_printoptions(precision=5, suppress=True)
from tqdm import tqdm

from common_3d.ops.farthest_point_sample.farthest_point_sample import farthest_point_sample_wrapper
from common_3d.utils.o3d_utils import np2pcd
from CompNet.utils.rot_utils import get_rotmat_from_filename


def convert_genre_to_canonical_space(genre_object_file, nb_pts=1024, verbose=False):
    if isinstance(genre_object_file, str):
        genre_object_file = Path(genre_object_file)
    assert genre_object_file.exists(), f'Could not find file {genre_object_file}'
    mesh = o3d.io.read_triangle_mesh(str(genre_object_file))
    vertices = np.asarray(mesh.vertices)
    # sample more points if not enough
    if len(vertices) < nb_pts:
        extra_verts = mesh.sample_points_uniformly(nb_pts*5) # pointcloud
        extra_verts = np.asarray(extra_verts.points)
        vertices = np.concatenate([vertices, extra_verts], axis=0)
    genre_pts = farthest_point_sample_wrapper(vertices, nb_pts=nb_pts)
    genre_pts -= np.mean(genre_pts, axis=0)
    if verbose:
        mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        genre_pcd = np2pcd(genre_pts)
        o3d.visualization.draw_geometries([mesh_frame, mesh, genre_pcd], window_name='genre space')

    cam_extrinsic = np.loadtxt('./CompNet/test_misc/cam_extrinsic.txt')
    R_W2C = cam_extrinsic[:3, :3]
    shape_id = Path(genre_object_file).stem
    R_M2W = get_rotmat_from_filename(shape_id)
    R = np.matmul(R_W2C, R_M2W)

    # convert GenRe to opencv camera space
    genre_pts_viewer = np.zeros_like(genre_pts)
    genre_pts_viewer[:, 0] = genre_pts[:, 2]
    genre_pts_viewer[:, 1] = -genre_pts[:, 1]
    genre_pts_viewer[:, 2] = genre_pts[:, 0]
    if verbose:
        genre_pcd_viewer = np2pcd(genre_pts_viewer)
        o3d.visualization.draw_geometries([mesh_frame, genre_pcd_viewer], window_name='opencv camera space')

    # convert GenRe to canonical space
    genre_pts_canonical = np.matmul(genre_pts_viewer, R)
    if verbose:
        genre_pcd_canonical = np2pcd(genre_pts_canonical)
        o3d.visualization.draw_geometries([mesh_frame, genre_pcd_canonical], window_name='canonical space')
    return genre_pts_canonical


def debug_sample():
    genre_base_dir = Path('/media/shanaf/HDD21/Songfang/project/partReconstruction/GenRe')
    model_class = 'chair'
    # test_obj_list = ['38012_rx317_ry36_rz0', '36061_rx351_ry286_rz0', '2786_rx16_ry351_rz0', '36387_rx0_ry328_rz0']
    shape_id = '2786_rx16_ry351_rz0'
    genre_object_file = genre_base_dir / f'genre_outputs_{model_class}' / f'{shape_id}.obj'

    genre_pts_canonical = convert_genre_to_canonical_space(genre_object_file, verbose=True)
    genre_pcd_canonical = np2pcd(genre_pts_canonical)
    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    o3d.visualization.draw_geometries([mesh_frame, genre_pcd_canonical], window_name='canonical space')

    # show ground truth
    gt_base_dir = '/media/shanaf/Data/data_v0'
    gt_pts_file = Path(gt_base_dir) / shape_id.split('_rx')[0] / 'point_sample' / 'pts-10000.txt'
    gt_pts_gt = np.loadtxt(gt_pts_file)
    gt_pcd_gt = np2pcd(gt_pts_gt)
    o3d.visualization.draw_geometries([mesh_frame, gt_pcd_gt, genre_pcd_canonical])


if __name__ == "__main__":
    debug_sample()
    # genre_base_dir = Path('/media/shanaf/HDD2/Songfang/project/partReconstruction/GenRe')
    # class_list = ['chair', 'bed', 'table', 'storagefurniture']
    # genre_pts_dir = genre_base_dir/'genre_1024_pts_canonical_space'
    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    #
    # for model_class in class_list:
    #     genre_class_dir = genre_base_dir / f'genre_outputs_{model_class}'
    #     obj_list = [x.stem for x in genre_class_dir.glob('*.obj')]
    #     genre_pts_class_dir = genre_pts_dir / model_class
    #     genre_pts_class_dir.mkdir(exist_ok=True, parents=True)
    #     for i in tqdm(range(len(obj_list))):
    #         obj_id = obj_list[i]
    #         genre_object_file = genre_class_dir/f'{obj_id}.obj'
    #         genre_pts_canonical = convert_genre_to_canonical_space(genre_object_file)
    #         genre_pcd_canonical = np2pcd(genre_pts_canonical)
    #         # o3d.visualization.draw_geometries([genre_pcd_canonical, mesh_frame],
    #         #                                   window_name=f'{obj_id}')
    #         out_genre_pts_file = genre_pts_class_dir/f'{obj_id}.txt'
    #         np.savetxt(str(out_genre_pts_file), genre_pts_canonical)







