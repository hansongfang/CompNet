import torch
from . import farthest_point_sample_cuda


class FarthestPointSampleFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, points, num_samples):
        index = farthest_point_sample_cuda.farthest_point_sample(points, num_samples)
        return index

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None,) * len(grad_outputs)


def farthest_point_sample(points, num_samples, transpose=True):
    """Farthest point sample.

    Args:
        points (torch.Tensor): (B, 3, N)
        num_samples (int): the number of points to sample
        transpose (bool): whether to transpose points. If false, then points should be (B, N, 3).

    Returns:
        torch.Tensor: (B, N), sampled indices.
    """
    if points.dim()==2:
        points = points.unsqueeze(0)
    if transpose:
        points = points.transpose(1, 2)
    points = points.contiguous()
    return FarthestPointSampleFunction.apply(points, num_samples)


@torch.no_grad()
def farthest_point_sample_wrapper(xyz, nb_pts):
    xyz = torch.tensor(xyz, dtype=torch.float32).cuda(non_blocking=True)
    fps_idx = farthest_point_sample(xyz, min(nb_pts, xyz.size(0)), transpose=False).squeeze(0)
    return xyz[fps_idx].cpu().numpy()
