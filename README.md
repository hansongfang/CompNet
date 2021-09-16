# Compositionally Generalizable 3D Structure Prediction

[comment]: <> (## Introduction)
In this work, We bring in the concept of compositional generalizability and factorizes the 3D shape reconstruction problem into proper sub-problems, each of which is tackled by a carefully designed neural sub-module with generalizability guarantee. Experiments on PartNet show that we achieve superior performance than baseline methods, which validates our problem factorization and network designs. [Link](https://arxiv.org/abs/2012.02493) to our paper. 

[comment]: <> (![Overview]&#40;./doc/teaser.jpg&#41;)

Check our YouTube videos below for more details.
[![PaperVideo](https://i.ytimg.com/vi_webp/a1Mghtz3erM/maxresdefault.webp)](https://youtu.be/a1Mghtz3erM) 

If you find this project useful for your research, please cite: 

```
@article{han2020compositionally,
author = {Han, Songfang and Gu, Jiayuan and Mo, Kaichun and Yi, Li and Hu, Siyu and Chen, Xuejin and Su, Hao},
title = {{C}ompositionally {G}eneralizable 3{D} {S}tructure {P}rediction},
journal = {arXiv preprint},
year = {2020}}
```

## How to use

### Installation
* Check out the source code 

    ```git clone https://github.com/hansongfang/CompNet.git && cd CompNet```
* Install dependencies 

    ```conda env create -f environment.yml  && conda activate CompNet```
* Compile CUDA extensions 

    ```cd common_3d && bash compile.sh```

## Training and evaluating 

Follow instructions in [CompNet README](https://github.com/hansongfang/CompNet/blob/main/CompNet/README.md)

## License

MIT Licence

## Updates

* [Sep 16, 2021] Preliminary version of Data and Code released.

