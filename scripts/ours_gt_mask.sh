#!/usr/bin/env bash

model_class=$1
echo $1

if [ "$1" = "bed" ]; then
  echo 'Process bed'
  python ./CompNet/test_all.py \
       --output_dir /media/shanaf/HDD21/Songfang/project/partReconstruction/ours/2020.11.23/output_gt_mask_debug3/bed \
       --render_dir './data/test/bed' \
       --shape_list './data/test/test_beds.txt' \
       --group_size \
       --choice 'rot group_size center' \
       --cfg_sizerelation './configs/SizeRelationNet.yaml' \
       --cfg_groupsize './configs/GroupAxisLengthNet.yaml' \
       --cfg_rot ./configs/RotNet.yaml \
       --cfg_center ./configs/JointNet.yaml \
       --thresh_size 0.9 \
#       --end_id 5

elif [ "$1" = "cabinet" ]; then
  echo 'Process cabinet'

  python ./CompNet/test_all.py \
     --output_dir /media/shanaf/HDD21/Songfang/project/partReconstruction/ours/2020.11.23/output_gt_mask_debug3/storagefurniture \
     --render_dir './data/test/storagefurniture' \
     --shape_list './data/test/test_storagefurnitures.txt' \
     --group_size \
     --choice 'rot group_size center' \
     --cfg_sizerelation './configs/SizeRelationNet.yaml' \
     --cfg_groupsize './configs/GroupAxisLengthNet.yaml' \
     --cfg_rot ./configs/RotNet.yaml \
     --cfg_center ./configs/JointNet.yaml \
     --thresh_size 0.9 \
#     --end_id 2

elif [ "$1" = "chair" ]; then
  echo 'Process chair'

  python ./CompNet/test_all.py \
     --output_dir /media/shanaf/HDD21/Songfang/project/partReconstruction/ours/2020.11.23/output_gt_mask_debug3/chair \
     --render_dir './data/test/chair' \
     --shape_list './data/test/test_chairs.txt' \
     --group_size \
     --choice 'rot group_size center' \
     --cfg_sizerelation './configs/SizeRelationNet.yaml' \
     --cfg_groupsize './configs/GroupAxisLengthNet.yaml' \
     --cfg_rot ./configs/RotNet.yaml \
     --cfg_center ./configs/JointNet.yaml \
     --thresh_size 0.9 \

elif [ "$1" = "table" ]; then
  echo 'Process table'

  python ./CompNet/test_all.py \
     --output_dir /media/shanaf/HDD21/Songfang/project/partReconstruction/ours/2020.11.23/output_gt_mask_debug3/table \
     --render_dir './data/test/table' \
     --shape_list './data/test/test_tables.txt' \
     --group_size \
     --choice 'rot group_size center' \
     --cfg_sizerelation './configs/SizeRelationNet.yaml' \
     --cfg_groupsize './configs/GroupAxisLengthNet.yaml' \
     --cfg_rot ./configs/RotNet.yaml \
     --cfg_center ./configs/JointNet.yaml \
     --thresh_size 0.9 \

fi
