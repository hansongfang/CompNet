# Part segmentation

## Installation

We use [Detectron2](https://github.com/facebookresearch/detectron2) for part segmentation.
The code is tested on `v0.2.1`.

```bash
# The latest fvcore is somehow incompatible to detectron==0.2.1
pip install fvcore==0.1.2.post20200912
pip install git+https://github.com/facebookresearch/detectron2.git@v0.2.1
```

## Data

First, please follow [CompNet README](../CompNet/README.md) to download data.
Then, run `link_data.sh` to link both train and test images.

To generate annotations for training on chairs:
```bash
python generate_annotation.py --root-dir datasets/images/train/chair --shape-list-fn datasets/images/train/chair/train_chairs.txt --output-path datasets/annotations/chair/instances_train.pkl
# Only used to monitor, not used for validation.
python generate_annotation.py --root-dir datasets/images/test/chair --shape-list-fn datasets/images/test/chair/test_chairs.txt --output-path datasets/annotations/chair/instances_test.pkl
```

## Train

`train_net.py` is simplified from `detectron2/tools/train_net.py`. Please refer to Detectron2 for details.

To train Mask R-CNN (ResNet50 FPN) for chairs with 2 GPUs:
```bash
python train_net.py --config-file configs/mask_rcnn_R_50_FPN_chair.yaml --num-gpus 2
```
The outputs are generated at `output/mask_rcnn_R_50_FPN_chair`, which can be modified by `OUTPUT_DIR` in the config.

## Inference

To predict masks for chairs with the trained model:
```bash
python predict_net.py --config-file configs/mask_rcnn_R_50_FPN_chair.yaml
```
By default, the predictions are generated at `datasets/predictions/test/${CLASS}`. The part masks of each shape is contained in a directory named by the shape id. The directory structure is similar to the training data.

To predict masks for other classes (e.g. bed):
```bash
python predict_net.py --config-file configs/mask_rcnn_R_50_FPN_chair.yaml --root-dir datasets/images/test/bed --output-dir datasets/predictions/test/bed
```

### Pretrained models

We also provide the pretrained model trained on chairs only.
Download the [pretrained model](https://drive.google.com/drive/folders/122kHzc01bF2pPMzMbP9xZWo0HhOgJd37?usp=share_link) and unzip to `./`.

To predict with pretrained weights:
```bash
python predict_net.py --config-file configs/mask_rcnn_R_50_FPN_chair.yaml --output-dir datasets/predictions_pretrained/test/chair --custom-weights --opts MODEL.WEIGHTS pretrained/mask_rcnn_R_50_FPN_chair/model_final.pth
```
