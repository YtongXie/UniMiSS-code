## Finetuning the UniMISS pretrained MiT on downstream 3D classification task

## Usage

### Data Preparation

* Download [MIDRC-RICORD-1A dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=80969742#80969742171ba531fc374829b21d3647e95f532c) and [MIDRC-RICORD-1B dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=80969771)
* Preprocess the data by re-spacing and save them in `dataset/RICORD_nii/`.

The folder structure of the dataset should be like

    dataset/RICORD_nii/
    ├── MIDRC-RICORD-1A
    |    └── ID419639-001155_date05-14-2008_605.000000-COR-3X3-96061.nii.gz
    |    └── ID440808-000016_date11-18-2006_605.000000-THORAX-PE-ART-AXIAL-3X3-8.812.nii.gz
    |    └── ...
    ├── MIDRC-RICORD-1B
    |    └── ID440808-000100_date05-31-2003_3.000000-ARTERIAL-AXIAL-THIN-14840.nii.gz
    |    └── ID419639-000824_date07-01-2002_3.000000-2.5mm-CAP-1.25mm-Neck-14456.nii.gz
    |    └── ...

### Training 
* cd UniMiSS-code/UniMiSS/Downstream/RICORD

* Download the pretrained models [UniMiss_small](https://drive.google.com/file/d/1YSMeIm9rAhVgivlvIZHUYjGS0-j2mm1M/view?) and put it into 'UniMiSS-code/UniMiSS/Downstream/RICORD/models/'.

* Run `python train.py -train_list='lists/RICORD_train.txt' -val_list='lists/RICORD_val.txt' -GPU='0' -NUM_CLASSES=2 -BATCH_SIZE=8 -EPOCH=200 -TRAIN_NUM=512 -LEARNING_RATE=0.00001 -optimizer='AdamW' -save_path='models/UniMiss_small/' -pre_train=True -pre_train_path='models/UniMiss_small.pth'` for training.

### Validation 
* Run `python test.py -test_list='lists/RICORD_test.txt' -GPU='0' -NUM_CLASSES=2 -BATCH_SIZE=8 -checkpoint_path='models/UniMiss_small'` for validation.


### Contact
Yutong Xie (yutong.xie678@gmail.com)
