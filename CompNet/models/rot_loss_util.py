import torch
import torch.nn.functional as F

from CompNet.utils.obb_utils import get_box_pts_torch, match_xydir


def get_unit_box_pts(device='cuda'):
    pts = torch.tensor(
        [[-1, -1, -1],
         [-1, 1, -1],
         [1, -1, -1],
         [1, 1, -1],
         [-1, -1, 1],
         [-1, 1, 1],
         [1, -1, 1],
         [1, 1, 1]],
        dtype=torch.float32, device=device)
    return pts * 0.5


def pairwise_distance_v2(x1, x2, p=2, normalize=False):
    dist = torch.norm(x1 - x2, p=p, dim=-1)
    if not normalize:
        dist = torch.pow(dist, p)
    return dist


def get_equal_rotations(device='cuda', sign=False):
    I = torch.eye(3, dtype=torch.float32, device=device)
    rotations = []
    for i in [0, 1, 2]:
        if sign:
            sign_i_list = [1., -1.]
        else:
            sign_i_list = [1.]
        for sign_i in sign_i_list:
            for j in [0, 1, 2]:
                if i == j:
                    continue
                sign_j_list = [1., -1.] if sign else [1.]
                for sign_j in sign_j_list:
                    xdir = I[:, i] * sign_i
                    ydir = I[:, j] * sign_j
                    zdir = torch.cross(xdir, ydir)
                    rotations.append(torch.stack([xdir, ydir, zdir], -1))
    return torch.stack(rotations, dim=0)


def rot_minn_emd_loss_v2(gt_box_size,
                         gt_box_rot,
                         pred_box_rot,
                         relative_size=False,
                         normalize=True,
                         reduce=False):
    if relative_size:
        gt_box_size = F.normalize(gt_box_size, p=2, dim=-1)

    eq_rots = get_equal_rotations(pred_box_rot.device, sign=True)  # [M, 3, 3]
    num_eq = eq_rots.size(0)

    unit_box_pts = get_unit_box_pts(pred_box_rot.device)  # [8, 3]
    size_box_pts = unit_box_pts.unsqueeze(0) * gt_box_size.unsqueeze(1)  # [B, 8, 3]
    gt_box_rots_eq = gt_box_rot.unsqueeze(1) @ eq_rots.unsqueeze(0)  # [B, M, 3, 3]
    gt_box_pts_eq = torch.matmul(size_box_pts.unsqueeze(1), gt_box_rots_eq.transpose(-1, -2))  # [B, M, 8, 3]
    pred_box_pts = torch.matmul(size_box_pts, pred_box_rot.transpose(-1, -2))  # [B, 8, 3]

    # emd
    pred_box_pts_expand = pred_box_pts.unsqueeze(1).repeat(1, num_eq, 1, 1)
    if normalize:
        emd_dist_all = pairwise_distance_v2(gt_box_pts_eq.reshape(-1, 8 * 3),
                                            pred_box_pts_expand.reshape(-1, 8 * 3),
                                            p=2,
                                            normalize=True)
        emd_dist_all = emd_dist_all.reshape(-1, num_eq)
    else:
        emd_dist_all = pairwise_distance_v2(gt_box_pts_eq.reshape(-1, 3),
                                            pred_box_pts_expand.reshape(-1, 3),
                                            p=2,
                                            normalize=False)
        emd_dist_all = emd_dist_all.reshape(-1, num_eq, 8).mean(dim=-1)
    emd_dist_min, _ = emd_dist_all.min(dim=-1)

    if reduce:
        emd_dist_min = torch.mean(emd_dist_min)
    return emd_dist_min


def rot_size_loss(pred_size, pred_xydir, gt_size, gt_xydir, cd_func, mode='train'):
    """
    Args:
        pred_size: torch.tensor, (batch_size, 3)
        pred_xydir: torch.tensor, (batch_size, 6)
        gt_size: torch.tensor, (batch_size, 3)
        gt_xydir: torch.tensor, (batch_size, 6)
        cd_func: chamfer distance function
        mode: string, either 'train', 'test' or 'val'

    Returns:
        cd_dist
    """
    gt_pts = get_box_pts_torch(size=gt_size,
                               rx=gt_xydir[:, :3],
                               ry=gt_xydir[:, 3:])
    pred_pts = get_box_pts_torch(size=pred_size,
                                 rx=pred_xydir[:, :3],
                                 ry=pred_xydir[:, 3:])
    dist1, dist2 = cd_func(gt_pts, pred_pts)

    if mode == 'test':
        cd_dist = torch.mean(dist1 + dist2, dim=-1)  # batch_size,  1
    else:
        cd_dist = torch.mean(dist1 + dist2)

    return cd_dist


def rot_match_size_loss(pred_xydir, gt_size, gt_xydir, cd_func, mode='train'):
    nn_inds = match_xydir(pred_xydir, gt_xydir)  # (b, 3)
    pred_size = torch.gather(gt_size, 1, nn_inds)

    cd_dist = rot_size_loss(pred_size,
                            pred_xydir,
                            gt_size,
                            gt_xydir,
                            cd_func,
                            mode=mode)
    return cd_dist
