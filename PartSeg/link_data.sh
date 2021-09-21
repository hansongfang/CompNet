#!/usr/bin/env bash

# Train
pushd $(pwd)
mkdir -p datasets/images/train && cd datasets/images/train
ln -s ../../../../data/train/chair chair
popd || exit

# Test
pushd $(pwd)
mkdir -p datasets/images/test && cd datasets/images/test
for CLASS in chair bed table storagefurniture;
do
    ln -s ../../../../data/test/$CLASS $CLASS
done
popd || exit