#!/usr/bin/env bash

model_class=$1

if [ "$1" = "chair" ]; then
  echo 'Process chair'

  python ./CompNet/test_all.py \
     --output_dir './outputs/prediction_predicted_mask/chair' \
     --render_dir './data/test_predicted_mask/chair' \
     --shape_list './data/test_predicted_mask/chair/test_chairs.txt' \
     --group_size \
     --choice 'rot group_size center' \
     --cfg_sizerelation './configs/SizeRelationNet.yaml' \
     --cfg_groupsize './configs/GroupAxisLengthNet.yaml' \
     --cfg_rot ./configs/RotNet.yaml \
     --cfg_center ./configs/JointNet.yaml \
     --thresh_size 0.9 \
#     --end_id 2

elif [ "$1" = "table" ]; then
  echo 'Process table'

  python ./CompNet/test_all.py \
     --output_dir './outputs/prediction_predicted_mask/table' \
     --render_dir './data/test_predicted_mask/table_by_chair' \
     --shape_list './data/test_predicted_mask/table_by_chair/test_tables.txt' \
     --group_size \
     --choice 'rot group_size center' \
     --cfg_sizerelation './configs/SizeRelationNet.yaml' \
     --cfg_groupsize './configs/GroupAxisLengthNet.yaml' \
     --cfg_rot ./configs/RotNet.yaml \
     --cfg_center ./configs/JointNet.yaml \
     --thresh_size 0.9 \
#     --end_id 2

elif [ "$1" = "bed" ]; then
  echo 'Process bed'

  python ./CompNet/test_all.py \
     --output_dir './outputs/prediction_predicted_mask/bed' \
     --render_dir './data/test_predicted_mask/bed_by_chair' \
     --shape_list './data/test_predicted_mask/bed_by_chair/test_beds.txt' \
     --group_size \
     --choice 'rot group_size center' \
     --cfg_sizerelation './configs/SizeRelationNet.yaml' \
     --cfg_groupsize './configs/GroupAxisLengthNet.yaml' \
     --cfg_rot ./configs/RotNet.yaml \
     --cfg_center ./configs/JointNet.yaml \
     --thresh_size 0.9 \
#     --end_id 2

elif [ "$1" = "cabinet" ]; then
  echo 'Process cabinet'

  python ./CompNet/test_all.py \
     --output_dir './outputs/prediction_predicted_mask/storagefurniture' \
     --render_dir './data/test_predicted_mask/storagefurniture_fs100' \
     --shape_list './data/test_predicted_mask/storagefurniture_fs100/test_storagefurnitures.txt' \
     --group_size \
     --choice 'rot group_size center' \
     --cfg_sizerelation './configs/SizeRelationNet.yaml' \
     --cfg_groupsize './configs/GroupAxisLengthNet.yaml' \
     --cfg_rot ./configs/RotNet.yaml \
     --cfg_center ./configs/JointNet.yaml \
     --thresh_size 0.9 \
#     --end_id 2

fi
