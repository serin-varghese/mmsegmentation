# STVFormer
STVFormer is a semantic segmentation framework built on top of OpenMMLab’s MMSegmentation. This repository extends MMSegmentation with a novel architecture—Spatial-Temporal Vision Transformer (STVFormer)—designed to enhance segmentation performance in video and sequential imagery by leveraging both spatial and temporal contextual cues.

This repo inherits MMSegmentation’s modular design, training pipelines, and evaluation tools, and integrates additional components specific to STVFormer, including custom backbones, temporal fusion modules, and dataset loaders optimized for sequential data.
d

![demo image](resources/seg_demo.gif)


## Installation

## Step1: Installation of mmsegmentation version. Forked from version mmsegmentation 1.2.2
Please refer to [get_started.md](docs/en/get_started.md#installation) for installation and [dataset_prepare.md](docs/en/user_guides/2_dataset_prepare.md#prepare-datasets) for dataset preparation.

## Step2: Preparation of datasets

### Cityscapes

Download the cityscapes dataset. Once you have logged in, your will have your username and password. For additional helpers and extended utilities, see:

- [city-scapes-script by cemsaz](https://github.com/cemsaz/city-scapes-script):
  A collection of scripts for automated downloading, extraction, and organization of the Cityscapes dataset. Useful for simplifying dataset preparation pipelines.

### BDD100k

TODO



## Training with STVFormer

```bash
python tools/train.py configs/stvformer/stvformer_cityscapes.py --work-dir work_dirs/stvformer_cityscapes
```
## Inference with STVFormer

For inference, we either use weights and biases or the
```bash
python tools/STV_Inference.py
```



## Contributing

We try to build upon MMSegmentation. Please refer to [CONTRIBUTING.md](.github/CONTRIBUTING.md) for the contributing guideline.
