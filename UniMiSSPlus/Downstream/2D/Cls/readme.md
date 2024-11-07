## Finetuning the UniMISS+ pretrained MiT+ on downstream 2D classification task

## Usage

### Data Preparation

* Download [CXR_Covid-19 Dataset](https://cxr-covid19.grand-challenge.org/), and put them in `dataset/`.

The folder structure of the dataset should be like

    dataset/CXR_Covid-19/CXR_Covid-19_Challenge/train
    ├── covid
    |    └── cov_1110.jpg
    |    └── cov_1111.jpg
    |    └── ...
    ├── normal
    |    └── normal-0000.jpg
    |    └── normal-0001.jpg
    |    └── ...
    ├── pneumonia
    |    └── pneumonia0000.jpg
    |    └── pneumonia0001.jpg
    |    └── ...

    dataset/CXR_Covid-19/CXR_Covid-19_Challenge/validation
    ├── covid
    |    └── cov_0.png
    |    └── cov_1.png
    |    └── ...
    ├── normal
    |    └── normal_0.png
    |    └── normal_1.png
    |    └── ...
    ├── pneumonia
    |    └── pneu_0.png
    |    └── pneu_1.png
    |    └── ...



### Run 
* cd UniMiSS-code/UniMiSSPlus/Downstream/2D/Cls

* Run `python main.py -train_list='dataset/CXR_Covid-19/CXR_Covid-19_Challenge_train_all.txt' -test_list='dataset/CXR_Covid-19/CXR_Covid-19_Challenge_test.txt' -GPU='0' -NUM_CLASSES=3 -BATCH_SIZE=32 -EPOCH=30 -deterministic=True -LEARNING_RATE=0.0001 -optimizer='AdamW' -save_path='models/UniMissPlus/' -pre_train=True -pre_train_path='../../../snapshots/UniMissPlus/UniMissPlus.pth'` for training and validation.


### Contact
Yutong Xie (yutong.xie678@gmail.com)
