# HRSAM: Efficiently Segment Anything in High-resolution Images

![Comparison of HRSAM and previous SOTA SegNext in high-precision interactive segmentation](assets/experiment-qualitative.png)
*Comparison of HRSAM and previous SOTA SegNext in high-precision interactive segmentation.*

## Installation

To install the required dependencies, follow the detailed instructions in the [Installation](./INSTALL.md).

## Datasets

To prepare the datasets, follow the detailed instructions in the [Datasets](./DATASETS.md).

## Pre-trained Models

We provide pre-trained models so that you can start testing immediately. You can download the pre-trained models and the corresponding configs from the following link:

[Pre-trained Models & Configs](https://mega.nz/file/kisziSBQ#Y1iiXF3kmeGgrjxToBN4lSS15vSL2KZ7GrasCUn1zQI)

The zip file contains the following directories:
- 'pretrain': the MAE-pretrained models
- 'work_dirs': contains the pretrained models and the corresponding configs


## Training the Model

To train the model, execute the following command:

e.g.
```bash
bash tools/dist_train.sh configs/hrsam/hqseg44k/hrsam_plusplus_simaug_1024x1024_bs1_40k.py 4
```

## Evaluating the Model

After training the model, you can evaluate it using the following command:

```bash
bash tools/dist_test_no_viz.sh configs/datasets_ext/eval_hqseq44k_val.py work_dirs/hrsam/hqseg44k/hrsam_plusplus_simaug_1024x1024_bs1_40k/iter_40000.pth 4
```
(the input resolution is 1024x1024)
or
```bash
bash tools/dist_test_no_viz.sh configs/datasets_ext/eval_hqseq44k_val.py work_dirs/hrsam/hqseg44k/hrsam_plusplus_simaug_1024x1024_bs1_40k/iter_40000.pth 4 -c configs/eval_custom/simseg_ts2048.py 
```
which will use 2048x2048 inputs in the evaluation.
