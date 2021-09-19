# Part segmentation

## Installation

We use [Detectron2]() for part segmentation.
The code is tested on `v0.2.1`.

```bash
pip install git+https://github.com/facebookresearch/detectron2.git@v0.2.1
```

## Data

First, please follow [CompNet README]() to download data.

```bash
# Link images directory
pushd $(pwd)
mkdir -p datasets/chair/images && cd datasets/chair/images
ln -s ../../../../data/train/chair train
ln -s ../../../../data/test/chair test
popd || exit

# Generate annotations
mkdir -p datasets/chair/annotations
python generate_annotation.py --root-dir datasets/chair/images/train --shape-list-fn datasets/chair/images/train/train_chairs.txt --output-path datasets/chair/annotations/instances_train.pkl
python generate_annotation.py --root-dir datasets/chair/images/test --shape-list-fn datasets/chair/images/test/test_chairs.txt --output-path datasets/chair/annotations/instances_test.pkl
```

## Train

