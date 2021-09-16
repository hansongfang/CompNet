"""
Utilities for Open3D (0.10.0)

Code modified from Jiayuan Gu.
"""
import numpy as np
import open3d as o3d


# ---------------------------------------------------------------------------- #
# Visualization
# ---------------------------------------------------------------------------- #
def np2pcd(points, colors=None, normals=None):
    """Convert numpy array to open3d PointCloud."""
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        colors = np.array(colors)
        if colors.ndim == 2:
            assert len(colors) == len(points)
        elif colors.ndim == 1:
            colors = np.tile(colors, (len(points), 1))
        else:
            raise RuntimeError(colors.shape)
        pc.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        assert len(points) == len(normals)
        pc.normals = o3d.utility.Vector3dVector(normals)
    return pc


def visualize_point_cloud(points, colors=None, normals=None,
                          show_frame=False, frame_size=1.0, frame_origin=(0, 0, 0)):
    """Visualize a point cloud."""
    pc = np2pcd(points, colors, normals)
    geometries = [pc]
    if show_frame:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=frame_size, origin=frame_origin)
        geometries.append(coord_frame)
    o3d.visualization.draw_geometries(geometries)


def draw_aabb(bbox, color=(0, 1, 0)):
    """Draw an axis-aligned bounding box."""
    assert len(bbox) == 6, f'The format of bbox should be xyzwlh, but received {len(bbox)}.'
    bbox = np.asarray(bbox)
    abb = o3d.geometry.AxisAlignedBoundingBox(bbox[0:3] - bbox[3:6] * 0.5, bbox[0:3] + bbox[3:6] * 0.5)
    abb.color = color
    return abb


def draw_obb(bbox, R, color=(0, 1, 0)):
    """Draw an oriented bounding box."""
    assert len(bbox) == 6, f'The format of bbox should be xyzwlh, but received {len(bbox)}.'
    obb = o3d.geometry.OrientedBoundingBox(bbox[0:3], R, bbox[3:6])
    obb.color = color
    return obb


# ---------------------------------------------------------------------------- #
# Computation
# ---------------------------------------------------------------------------- #
def compute_normals(points, search_param=None, camera_location=(0.0, 0.0, 0.0)):
    """Compute normals."""
    pcd = np2pcd(points)
    if search_param is None:
        pcd.estimate_normals()
    else:
        pcd.estimate_normals(search_param=search_param)
    pcd.orient_normals_towards_camera_location(camera_location)
    normals = np.array(pcd.normals)
    return normals


def voxel_down_sample(points, voxel_size, min_bound=(-5.0, -5.0, -5.0), max_bound=(5.0, 5.0, 5.0)):
    """Downsample the point cloud and return sample indices."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    downsample_pcd, mapping, index_buckets = pcd.voxel_down_sample_and_trace(
        voxel_size, np.array(min_bound)[:, None], np.array(max_bound)[:, None])
    sample_indices = [int(x[0]) for x in index_buckets]
    return sample_indices


# ---------------------------------------------------------------------------- #
# Unit test
# ---------------------------------------------------------------------------- #
def test_compute_normals():
    mesh = o3d.geometry.TriangleMesh.create_box(1.0, 1.0, 1.0)
    mesh.translate(np.array([-0.5, -0.5, -0.5])[:, None])
    pcd = mesh.sample_points_uniformly(number_of_points=500)
    points = np.array(pcd.points)
    normals = compute_normals(points)
    visualize_point_cloud(points, normals=normals, show_frame=True)
