# Datasets

### COCO
To acquire the COCO dataset, please visit [cocodataset](https://cocodataset.org/#download). The following files are required: [2017 Train Images](http://images.cocodataset.org/zips/train2017.zip), [2017 Val Images](http://images.cocodataset.org/zips/val2017.zip), and [2017 Panoptic Train/Val Annotations](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip). These should be downloaded into the `data` directory.

Alternatively, the dataset can be downloaded using the provided script:
```shell
cd data/coco2017
bash coco2017.sh
```
The data is organized as follows:
```
data/coco2017/
├── annotations
│   ├── panoptic_train2017 [118287 entries exceeds filelimit, not opening dir]
│   ├── panoptic_train2017.json
│   ├── panoptic_val2017 [5000 entries exceeds filelimit, not opening dir]
│   └── panoptic_val2017.json
├── coco2017.sh
├── train2017 [118287 entries exceeds filelimit, not opening dir]
└── val2017 [5000 entries exceeds filelimit, not opening dir]
```


### LVIS
The LVIS dataset can be downloaded by visiting [lvisdataset](https://www.lvisdataset.org/dataset). Here, you'll find both the images and annotations.

The data is organized as follows:
```
data/lvis/
├── lvis_v1_train.json
├── lvis_v1_train.json.zip
├── lvis_v1_val.json
├── lvis_v1_val.json.zip
├── train2017 [118287 entries exceeds filelimit, not opening dir]
├── train2017.zip
├── val2017 [5000 entries exceeds filelimit, not opening dir]
└── val2017.zip
```


### DAVIS
Please download [DAVIS](https://drive.google.com/file/d/1-ZOxk3AJXb4XYIW-7w1-AXtB9c8b3lvi/view?usp=sharing) from [FocusCut](https://github.com/frazerlin/focuscut)

The data is organized as follows:
```
data/
├── davis
│   └── DAVIS
│       ├── gt [345 entries exceeds filelimit, not opening dir]
│       ├── img [345 entries exceeds filelimit, not opening dir]
│       └── list
│           ├── val_ctg.txt
│           └── val.txt
```

## HQSeg44K
Please refer to [SAM-HQ Repository](https://github.com/SysCV/sam-hq/blob/main/train/README.md) for more information on the HQSeg44K dataset.

The data files are organized as follows:

```shell
data/sam-hq
├── cascade_psp
│   ├── cascade_psp.zip
│   ├── DUTS-TE [10038 entries exceeds filelimit, not opening dir]
│   ├── DUTS-TR [21106 entries exceeds filelimit, not opening dir]
│   ├── ecssd [2000 entries exceeds filelimit, not opening dir]
│   ├── fss_all [20006 entries exceeds filelimit, not opening dir]
│   └── MSRA_10K [20000 entries exceeds filelimit, not opening dir]
├── DIS5K
│   ├── DIS5K Dataset Terms of Use.pdf
│   ├── DIS-TR
│   │   ├── gt [3000 entries exceeds filelimit, not opening dir]
│   │   └── im [3000 entries exceeds filelimit, not opening dir]
│   └── DIS-VD
│       ├── gt [470 entries exceeds filelimit, not opening dir]
│       └── im [470 entries exceeds filelimit, not opening dir]
├── DIS5K.zip
├── hqseg44k_ignore_prefix.json
└── thin_object_detection
    ├── COIFT
    │   ├── 20240127_093512_analysis.txt
    │   ├── images [280 entries exceeds filelimit, not opening dir]
    │   └── masks [280 entries exceeds filelimit, not opening dir]
    ├── HRSOD
    │   ├── 20240127_094055_analysis.txt
    │   ├── images [287 entries exceeds filelimit, not opening dir]
    │   └── masks_max255 [287 entries exceeds filelimit, not opening dir]
    ├── ThinObject5K
    │   ├── images_test [500 entries exceeds filelimit, not opening dir]
    │   ├── images_train [4748 entries exceeds filelimit, not opening dir]
    │   ├── masks_test [500 entries exceeds filelimit, not opening dir]
    │   └── masks_train [4748 entries exceeds filelimit, not opening dir]
    └── thin_object_detection.zip
```
