## BDD100K

- You could download BDD100k datasets from  [here](https://dl.cv.ethz.ch/bdd100k/data/). [Note: There are issues with accessing the [official download page](https://bdd-data.berkeley.edu/)]. Use the [official github repo](https://github.com/bdd100k/bdd100k) for additional tools.

- Download the 10k_images_train, 10k_images_test and 10k_images_val zips.

- After download, unzip by the following instructions:

  ```bash
  unzip ~/bdd100k_images_10k.zip -d ~/mmsegmentation/data/
  unzip ~/bdd100k_sem_seg_labels_trainval.zip -d ~/mmsegmentation/data/
  ```

```none
mmsegmentation
├── mmseg
├── tools
├── configs
├── data
│   ├── bdd100k
│   │   ├── images
│   │   │   └── 10k
|   │   │   │   ├── test
|   │   │   │   ├── train
|   │   │   │   └── val
│   │   └── labels
│   │   │   └── sem_seg
|   │   │   │   ├── colormaps
|   │   │   │   │   ├──train
|   │   │   │   │   └──val
|   │   │   │   ├── masks
|   │   │   │   │   ├──train
|   │   │   │   │   └──val
|   │   │   │   ├── polygons
|   │   │   │   │   ├──sem_seg_train.json
|   │   │   │   │   └──sem_seg_val.json
|   │   │   │   └── rles
|   │   │   │   │   ├──sem_seg_train.json
|   │   │   │   │   └──sem_seg_val.json
```
