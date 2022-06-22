'''
get the evaluation emd result 

The file is organized version of some part of the notebook
to extract points and compute emd between predicted and GT PCD
'''
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import os.path as osp
import numpy as np
import math
import argparse
from tqdm import tqdm
import open3d as o3d
import pandas as pd
from termcolor import colored
import torch
import pickle

from CompNet.utils.rot_utils import get_rotmat_from_filename
from CompNet.test_misc.obb2verts_misc import sample_obb_list_verts
from common_3d.ops.farthest_point_sample.farthest_point_sample import farthest_point_sample_wrapper
from common_3d.ops.emd.emd_module import emdModule
emd_dist = emdModule()

def parse_args():
    parser = argparse.ArgumentParser(description='After running test_all.py, run this program'
        " to get the evaluation result. \n"
        "This program contains two steps. First generate the pcd from the predicted cuboid"
        " from test output and the gt pcd from gt visible cuboids or surface points, if you"
        " have not done so. \n"
        "Then compute the EMD distance between those two sets of pcd's. \n"
        "Modify the value in the Main block and then run."
        )
    # parser.add_argument(
    #     "--cfg",
    #     dest="config_file",
    #     default="",
    #     help="path to config file",
    #     type=str,
    # )
    # parser.add_argument('--convert', action='store_true'
    #     help="",
    # )

    args = parser.parse_args()
    return args

def convert_pred_to_canonical_space(obb_file, shape_id=None, visible_pid_list=None):
    """ Convert obb file to PCD of size 1024 in canonical space"""
    obb_file = Path(obb_file)
    assert obb_file.exists(), f'Not found file {obb_file}'

    obb_dict = np.load(str(obb_file), allow_pickle=True).item()
    if visible_pid_list is None:
        obb_list = [item for key, item in obb_dict.items()]
    else:
        obb_list = [item for key, item in obb_dict.items() if key in visible_pid_list]
    obb_pts = sample_obb_list_verts(obb_list, nb_pts=1024)
    obb_pts -= np.mean(obb_pts, axis=0)

    if shape_id is None:
        shape_id = obb_file.parent.stem
    rotmat = get_rotmat_from_filename(shape_id)
    R = rotmat

    obb_pts_canonical_space = np.matmul(obb_pts, R)

    return obb_pts_canonical_space


def emd_np_warpper(pts1, pts2, verbose=False):
    if verbose:
        pcd1 = np2pcd(pts1)
        pcd2 = np2pcd(pts2)
        o3d.visualization.draw_geometries([pcd1, pcd2])
    batch_pts1 = torch.tensor(pts1).float().cuda().unsqueeze(0)
    batch_pts2 = torch.tensor(pts2).float().cuda().unsqueeze(0)
    dist12, assigment = emd_dist(batch_pts1, batch_pts2, 0.05, 3000)
    return float(torch.mean(dist12))


def eval_emd_from_obb_in_npy(pred_base_dir,
                             gt_base_dir,
                             out_stat_dir,
                            class_list=['chair', 'bed', 'table', 'storagefurniture'],
                            norm_scale=False,
                            result_file_suffix=''):
    """
    Convert both pred & gt obb in .npy files to caonical 1024 pts using FPS,
    and then calculate emd for each class.
    """
    pred_base_dir = Path(pred_base_dir)
    # out_pts_dir = Path(out_pts_dir)
    gt_base_dir = Path(gt_base_dir)
    cls_dict = {}

    for model_class in class_list:
        pred_class_dir = pred_base_dir / model_class / "center"
        gt_class_dir = gt_base_dir / model_class
        obj_list = [x.stem for x in pred_class_dir.iterdir() if x.is_dir()]
        print(f"Calculating average emd for {model_class}, num of obj: {len(obj_list)}")

        all_emd_dist, save_list = [], [] # record emd dist and all relevant info
        fail_dict = {}                   # record failing obj
        for i in tqdm(range(len(obj_list))):
            obj_id = obj_list[i]
            pred_obb_file = pred_class_dir / f'{obj_id}' / f'{obj_id}.npy'
            gt_obb_file = gt_class_dir /  f'{obj_id}'/ 'gt.npy'
            # gt_obb_file = gt_class_dir /  f'{obj_id}.txt'
            
            if not pred_obb_file.exists():
                fail_dict[obj_id] = f"Cannot find pred obb file: {pred_obb_file}"
                continue
            if not gt_obb_file.exists():
                fail_dict[obj_id] = f"Cannot find gt obb file: {gt_obb_file}"
                continue
                
            # in model space
            pred_pts = convert_pred_to_canonical_space(pred_obb_file, shape_id=obj_id);
            gt_pts = convert_pred_to_canonical_space(gt_obb_file, shape_id=obj_id);
            # gt_pts = np.loadtxt(str(gt_obb_file))
            
            if len(pred_pts) != 1024:
                fail_dict[obj_id] = "pred pcd point count does not match"
                continue
            if len(gt_pts) != 1024:
                fail_dict[obj_id] = "gt pcd point count does not match"
                continue
            if math.isnan(pred_pts[0, 0]):
                fail_dict[obj_id] = "pred pcd [0,0] is nan"
                continue
            
            gt_pts -= np.mean(gt_pts, axis=0)
            pred_pts -= np.mean(pred_pts, axis=0)
            if norm_scale:
                gt_bound = np.sqrt(np.max(np.sum(gt_pts ** 2, axis=1)))
                pred_bound = np.sqrt(np.max(np.sum(pred_pts ** 2, axis=1)))
                if pred_bound < 0.05:
                    fail_dict[obj_id] = "pred max norm less than 0.05"
                    continue
                scale = gt_bound / pred_bound
                pred_pts_v2 = pred_pts * scale
            else:
                pred_pts_v2 = pred_pts

            dist = emd_np_warpper(pred_pts_v2, gt_pts, verbose=False)
            if math.isnan(dist):
                fail_dict[obj_id] = "Calculated emd dist is nan"
                continue
            all_emd_dist.append(dist)
            save_list.append([obj_id, model_class, dist, "with-scale" if norm_scale else "wo-scale"])
            
        if len(all_emd_dist) == 0:
            print(f'Class {model_class}, empty list')
            print(f'out_fail_file:', fail_dict)
        else:
            cls_dict[model_class] = np.mean(all_emd_dist)
            print(colored(f'Class {model_class}, {np.mean(all_emd_dist)}', on_color="on_cyan"))
            
            # save all dist as a pd table
            if_scale = "with-scale" if norm_scale else "wo-scale"
            out_res_file = pred_base_dir / f'emd_{model_class}_{if_scale}{result_file_suffix}.pkl'
            df = pd.DataFrame(save_list, columns=['obj_id', 'model_class', 'dist', 'scale'])
            df.to_pickle(str(out_res_file))
            print(f'Save result to {out_res_file}')
            
            # save failure info
            out_fail_file = pred_base_dir / f'emd_{model_class}_{if_scale}_failure{result_file_suffix}.pkl'
            with open(out_fail_file, 'wb') as f:
                pickle.dump(fail_dict, f)
            print(f'Failed: {len(fail_dict)}, Save failure info to {out_fail_file}')
            print(f'fail_dict:', fail_dict)
    print(cls_dict)
    

def sample_pcd_from_pred_obb(pred_base_dir, out_pts_base_dir, class_list:list=['chair', 'bed', 'table', 'storagefurniture']):
    """
    Sample canonical pcd from predicted obb file by the center module.
    Save output PCD as .txt file in the result folder of `pred_base_dir`.
    """
    pred_base_dir = Path(pred_base_dir)
    out_pts_base_dir = Path(out_pts_base_dir)
    for model_class in class_list:
        pred_class_dir = pred_base_dir / model_class / "center"
        out_pts_class_dir = out_pts_base_dir / model_class / "result"
        out_pts_class_dir.mkdir(parents=True, exist_ok=True)
        print(f"Save predicted {model_class} PCD to '{out_pts_class_dir}'")
        
        obj_list = [x.stem for x in pred_class_dir.iterdir() 
                    if x.is_dir() and x.match("*_rx*_ry*_rz*")]
        for obj_id in tqdm(obj_list):
            pred_obb_file = pred_class_dir / f'{obj_id}' / f'{obj_id}.npy'
            obb_pts_canonical = convert_pred_to_canonical_space(pred_obb_file, shape_id=obj_id);
            
            out_pts_file = out_pts_class_dir / f'{obj_id}.txt'
            np.savetxt(str(out_pts_file), obb_pts_canonical)
        
def sample_pcd_from_gt_obb(gt_base_dir, out_pts_base_dir, class_list:list=['chair', 'bed', 'table', 'storagefurniture']):
    """
    Sample canonical pcd from gt obb file. 
    Note that only visible cuboids are used.
    Save output PCD as .txt file in `out_pts_base_dir`.
    """
    gt_base_dir = Path(gt_base_dir)
    out_pts_base_dir = Path(out_pts_base_dir)
    for model_class in class_list:
        gt_class_dir = gt_base_dir / model_class
        out_pts_class_dir = out_pts_base_dir / model_class
        out_pts_class_dir.mkdir(parents=True, exist_ok=True)
        print(f"Save visible GT {model_class} PCD to '{out_pts_class_dir}'")
        
        obj_list = [x.stem for x in gt_class_dir.iterdir() 
                    if x.is_dir() and x.match("*_rx*_ry*_rz*")]
        for obj_id in tqdm(obj_list):
            # only sample from visible cuboids
            gt_obb_file = gt_class_dir / obj_id / 'gt.npy'
            obj_file = gt_class_dir / obj_id / f'{obj_id}.npy'
            visible_pid_list = np.load(obj_file, allow_pickle=True).item()['vis_parts'] 
            obb_pts_canonical = convert_pred_to_canonical_space(gt_obb_file, shape_id=obj_id,
                                                               visible_pid_list=visible_pid_list);
            
            out_pts_file = out_pts_class_dir / f'{obj_id}.txt'
            np.savetxt(str(out_pts_file), obb_pts_canonical)

def sample_pcd_from_gt_surface_pcd(gt_base_dir, gt_surface_pts_base_dir, out_pts_base_dir, class_list:list=['chair', 'bed', 'table', 'storagefurniture']):
    """
    Sample canonical pcd from 10000 points from gt surface in `gt_surface_pts_base_dir`.
    Use gt_base_dir to get the obj_id list.
    Save output PCD as .txt file in `out_pts_base_dir`.
    """
    gt_base_dir = Path(gt_base_dir)
    gt_surface_pts_base_dir = Path(gt_surface_pts_base_dir)
    out_pts_base_dir = Path(out_pts_base_dir)
    for model_class in class_list:
        gt_class_dir = gt_base_dir / model_class
        out_pts_class_dir = out_pts_base_dir / model_class
        out_pts_class_dir.mkdir(parents=True, exist_ok=True)
        print(f"Save surface GT {model_class} PCD to '{out_pts_class_dir}'")
        
        obj_list = [x.stem for x in gt_class_dir.iterdir() 
                    if x.is_dir() and x.match("*_rx*_ry*_rz*")]
        for obj_id in tqdm(obj_list):
            anno_id = obj_id.split("_")[0]
            gt_surface_pts_file = gt_surface_pts_base_dir / anno_id / "point_sample" / "ply-10000.ply"
            # FPS from 10000 presampled gt surface pts
            assert gt_surface_pts_file.exists(), f"Cannot find: {str(gt_surface_pts_file)}"
            pcd = o3d.io.read_point_cloud(str(gt_surface_pts_file))
            gt_surface_pts = np.asarray(pcd.points)
            gt_surface_pts = farthest_point_sample_wrapper(gt_surface_pts, 1024)
            
            out_pts_file = out_pts_class_dir / f'{obj_id}.txt'
            np.savetxt(str(out_pts_file), gt_surface_pts)

def eval_emd_from_pcd(pred_pts_base_dir,
                      gt_pts_base_dir,
                      out_stat_dir,
                      class_list=['chair', 'bed', 'table', 'storagefurniture'],
                      norm_scale=False,
                      result_file_suffix=''):
    """
    Directly alculate emd for each class based on sampled predicted and gt PCD beforehand. 
    """
    pred_pts_base_dir = Path(pred_pts_base_dir)
    gt_pts_base_dir = Path(gt_pts_base_dir)
    out_stat_dir = Path(out_stat_dir)
    out_stat_dir.mkdir(parents=True, exist_ok=True)
    
    cls_dict = {"result":{}, "failed":{}}
    if_scale = "with-scale" if norm_scale else "wo-scale"

    for model_class in class_list:
        pred_class_dir = pred_pts_base_dir / model_class / "result"
        gt_class_dir = gt_pts_base_dir / model_class
        obj_list = [x.stem for x in pred_class_dir.iterdir() if x.match("*_rx*_ry*_rz*")]
        print(f"Calculating average emd for {model_class}, num of obj: {len(obj_list)}")

        all_emd_dist, save_list = [], [] # record emd dist and all relevant info
        fail_dict = {}                   # record failing obj
        for i in tqdm(range(len(obj_list))):
            obj_id = obj_list[i]
            pred_pts_file = pred_class_dir / f'{obj_id}.txt'
            gt_pts_file = gt_class_dir /  f'{obj_id}.txt'
            
            if not pred_pts_file.exists():
                fail_dict[obj_id] = f"Cannot find pred obb file: {pred_pts_file}"
                continue
            if not gt_pts_file.exists():
                fail_dict[obj_id] = f"Cannot find gt obb file: {gt_pts_file}"
                continue
                
            # in model space
            pred_pts = np.loadtxt(str(pred_pts_file))
            gt_pts = np.loadtxt(str(gt_pts_file))
            
            if len(pred_pts) != 1024:
                fail_dict[obj_id] = "pred pcd point count does not match"
                continue
            if len(gt_pts) != 1024:
                fail_dict[obj_id] = "gt pcd point count does not match"
                continue
            
            gt_pts -= np.mean(gt_pts, axis=0)
            pred_pts -= np.mean(pred_pts, axis=0)
            if norm_scale:
                gt_bound = np.sqrt(np.max(np.sum(gt_pts ** 2, axis=1)))
                pred_bound = np.sqrt(np.max(np.sum(pred_pts ** 2, axis=1)))
                if pred_bound < 0.05:
                    fail_dict[obj_id] = "pred max norm less than 0.05"
                    continue
                scale = gt_bound / pred_bound
                pred_pts_v2 = pred_pts * scale
            else:
                pred_pts_v2 = pred_pts

            dist = emd_np_warpper(pred_pts_v2, gt_pts, verbose=False)
            if math.isnan(dist):
                fail_dict[obj_id] = "Calculated emd dist is nan"
                continue
            all_emd_dist.append(dist)
            save_list.append([obj_id, model_class, dist, if_scale])
            
        if len(all_emd_dist) == 0:
            print(f'Class {model_class}, empty list')
            print(f'out_fail_file:', fail_dict)
        else:
            cls_dict["result"][model_class] = np.mean(all_emd_dist)
            print(colored(f'Class {model_class}, {np.mean(all_emd_dist)}', on_color="on_cyan"))
            
            # save all dist as a pd table
            out_res_file = out_stat_dir / f'emd_{model_class}_{if_scale}{result_file_suffix}.pkl'
            df = pd.DataFrame(save_list, columns=['obj_id', 'model_class', 'dist', 'scale'])
            df.to_pickle(str(out_res_file))
            print(f'Save result to {out_res_file}')
            
            # save failure info
            print(f'Failed: {len(fail_dict)}')
            print(f'fail_dict:', fail_dict)
            cls_dict["failed"][model_class] = fail_dict
    
    out_stat_file = out_stat_dir / f'result_{"".join(class_list)}_{if_scale}{result_file_suffix}.pkl'
    with open(str(out_stat_file), 'wb') as f:
        pickle.dump(cls_dict, f)
    return cls_dict

def main():
    args = parse_args()
#     pass

if __name__ == "__main__":
    main()

    ### Define pred and gt dir
    ## model output dir
    pred_base_dir = '/q5chen-cephfs/project_file/testComp/CompNet/outputs/prediction_predicted_mask'
    ## path to store sampled pred pcd
    pred_out_pts_base_dir = "/q5chen-cephfs/project_file/testComp/CompNet/outputs/prediction_predicted_mask"

    ## base dir for gt shape, containing surface pts for each shape
    gt_surface_pts_base_dir = "/sf-fast2/DATA/data_v0"                     
    ## base dir for gt rendering, containing gt cuboids and visible id lists for each class and shape
    gt_base_dir = "/q5chen-cephfs/project_file/testComp/CompNet/data/test" 
    ## path to store sampled gt pcd
    gt_out_pts_base_dir = "/q5chen-cephfs/project_file/testComp/CompNet/data/GT_sampled_pcd/partnet_1024_pts" # old GT surface pts

    # class_list=['bed', 'chair', 'table', 'storagefurniture']
    class_list = ["chair"]

    ### Step 1a: Sample pred pts if you haven't done so
    # sample_pcd_from_pred_obb(pred_base_dir, out_pts_base_dir=pred_out_pts_base_dir, class_list=class_list)

    ### Step 1b: Sample GT pts if you haven't done so
    # sample_pcd_from_gt_obb(gt_base_dir, out_pts_base_dir=gt_out_pts_base_dir, class_list=class_list)
    # sample_pcd_from_gt_surface_pcd(gt_base_dir, gt_surface_pts_base_dir, out_pts_base_dir=gt_out_pts_base_dir, class_list=class_list)

    ### Step 2: Calculate emd based on sampled pred and gt pcd
    # out_stat_dir = "/q5chen-cephfs/project_file/testComp/CompNet/outputs/prediction_predicted_mask/result_stat"
    out_stat_dir = osp.join(pred_base_dir, "result_stat")
    eval_emd_from_pcd(pred_pts_base_dir=pred_out_pts_base_dir, gt_pts_base_dir=gt_out_pts_base_dir, out_stat_dir=out_stat_dir,
                    class_list=class_list, norm_scale=False)