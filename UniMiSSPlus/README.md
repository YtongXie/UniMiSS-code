# UniMiSS+ code
This is the official pytorch implementation of our extended IEEE-TPAMI paper ["UniMiSS+: Universal Medical Self-Supervised Learning From Cross-Dimensional Unpaired Data"](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10617802). 

<div align="center">
  <img width="80%" alt="DINO illustration" src="UniMiSSPlus.png">
</div>

## Requirements
CUDA 11.1<br />
Python 3.7<br /> 
Pytorch 1.8.1<br /> 
Torchvision 0.9.1<br />
matplotlib==3.1.2<br />
nibabel==3.2.1<br />
opencv-python==4.4.0.46<br />
batchgenerators==0.19.7<br />
typing-extensions==4.2.0<br />


## Usage
### Data Preparation
* cd UniMiSSPlus/data
* Download [DeepLesion data](https://nihcc.app.box.com/v/DeepLesion)
* Save to 3D nifti volumes by [DL_save_nifti.py](https://nihcc.app.box.com/v/DeepLesion), and then put them into `3D_images`
* Run `python extract_subvolumes.py` to extract sub-volumes, and put them into `3D_subvolumes`
* The image folder of 3D images should be like:

```.python
    data/3D_subvolumes/
    ├── 000001_01_01_103-115_dep0.nii.gz
    ├── 000001_01_01_103-115_dep1.nii.gz
    ├── 000001_01_01_103-115_dep2.nii.gz
    ├── 000001_01_01_103-115_dep3.nii.gz
    ├── 000001_01_01_103-115_dep4.nii.gz
    ├── 000001_02_01_008-023_dep0.nii.gz
    ├── 000001_02_01_008-023_dep1.nii.gz
    ├── 000001_02_01_008-023_dep2.nii.gz
    ├── 000001_02_01_008-023_dep3.nii.gz
    ├── 000001_02_01_008-023_dep4.nii.gz
    ├── 000001_02_01_008-023_dep5.nii.gz

```
        
* Download [NIH ChestX-ray8 dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
* Resize ChestX-ray8 images into 512×512 and put the resized images into `2D_images`.
* The image folder of ChestX-ray8 should look like this:

```.python
    data/2D_images/
    ├── 00000001_000.png
    ├── 00000001_001.png
    ├── ...
```

* Run `python listSSL.py` to generate `3D_images.txt` and `2D_images.txt`.

Then, use [pycuda_drr](https://github.com/yuta-hi/pycuda_drr) to generate DRR images
* cd UniMiSSPlus/pycuda_drr
* Run `python setup.py install` to install `pycuda_drr`
* Run `python rendering_DL.py` to generate DRR images

### Pre-training 
* cd UniMiSSPlus
* Run `sh run_UniMissPlus.sh` for self-supervised pre-training.

### Pretrained weights 
* Pretrained models are available in [UniMissPlus](https://drive.google.com/file/d/1WwnmpFX_7Q0Ec7CdxRJPniLtFKg5HwU0/view?usp=sharing).

### Finetuning 
* Finetuning codes are available in [Downstream](https://github.com/YtongXie/UniMiSS-code/tree/main/UniMiSSPlus/Downstream).

### Acknowledgements
Part of codes is reused from the [DINO](https://github.com/facebookresearch/dino) and [pycuda_drr](https://github.com/yuta-hi/pycuda_drr). 
Thanks to authors for the codes of DINO and pycuda_drr.

### Contact
Yutong Xie (yutong.xie678@gmail.com)

