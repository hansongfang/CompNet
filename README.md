# Compositionally Generalizable 3D Structure Prediction

## Branch Description:
This branch contains some evaluation & experiment code for this project. Most code are contained in a notebook `experiments.ipynb` with proper titles and separations to divide each part of the experiments. 

Here are some descriptions of each section:

1. Check each kind of GT and output file: 
    * just visualize all posible files to use
2. Visualize/debug HIS result on gt json, gt visible json, and pred json:
    * Compare the tree structure in the json file predicted by HIS with GT tree structure.
3. HIS with no hierarchy:
    * Test our new part matching metric that is similar to HIS algorithm but without the part hierarchy.
4. Part bbox IoU:
    * Test another part matching metric: compute the average of all part bbox IoU in world space after Hungarian matching using the part IoU as the metrics.
    * Under this approach, we tried: linear_assignment_iou, linear_assignment_iou_weighed_by_vol,linear_assignment_iou_plus_unmatched_max_iou, linear_assignment_iou_plus_unmatched_zero, and chamfer_like_iou
5. Part prediction F1 Score:
    * Test another part matching metric: Compute the F1 score of part mataching with a chamfer dist threshold as the passing criteria and the Hungarian matching using the Chamfer dist in canonical space
    * Under this approach, we tried: ave_part_min_chamfer_dist, ave_part_matched_chamfer_dist, part_pred_f1, matched_part_pred_f1
    * Contain code to plot multiple 3d graphs at once
6. Sample pcd and Calculated emd between pred and gt:
    * Contain code to sample point cloud from gt and different models and compute the EMD between the sample PCD and the GT PCD.

Each branch may contain the code of the algorithm and the main code to run on the data. You can refer to the main code to prepare your data and build similar directory structure.
