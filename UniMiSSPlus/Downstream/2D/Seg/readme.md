## Finetuning the UniMISS+ pretrained MiT+ on downstream 2D segmentation task

## Usage

### Data Preparation

* Download [JSTR Dataset](https://drive.google.com/file/d/1ifMkZlikz34pMeuknPtnCdG1m5zX-rb_/view?usp=sharing), and put them in `dataset/`.

The folder structure of the dataset should be like

    dataset/Images/
    ├── JPCLN001.png
    ├── JPCLN002.png
    |   ...
    dataset/Masks/
    ├── clavicle
    |    └── JPCLN001.gif
    |    └── JPCLN002.gif
    |    └── ...
    ├── heart
    |    └── JPCLN001.gif
    |    └── JPCLN002.gif
    |    └── ...
    | ...


### Run 
* cd UniMiSS-code/UniMissPlus/Downstream/2D/Seg

* Run `python main.py -train_list='dataset/Training_seg_all.txt' -test_list='dataset/Test_seg.txt' -GPU='0' -NUM_CLASSES=3 -BATCH_SIZE=32 -EPOCH=200 -TRAIN_NUM=1600 -deterministic=True -LEARNING_RATE=0.0001 -optimizer='AdamW' -save_path='models/UniMissPlus/' -pre_train=True -pre_train_path='../../../snapshots/UniMissPlus/UniMissPlus.pth'` for training and validation.


### Contact
Yutong Xie (yutong.xie678@gmail.com)
