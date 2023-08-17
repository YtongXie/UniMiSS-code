## Finetuning the UniMISS pretrained MiT on downstream 3D segmentation tasks

## Usage

### 0. Installation

* Install nnUNet and MiTnnu as below
  
```
Download nnUNet from the link https://github.com/YtongXie/CoTr/tree/main/nnUNet.

cd nnUNet
pip install -e .

cd UniMiSS-code/UniMiSS/Downstream/MiTnnu
pip install -e .
```

### 1. Data Preparation

** Take the BCV dataset as an example,
* Download [BCV dataset](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)
* Preprocess the BCV dataset according to the uploaded nnUNet package.

### 2. Training 
cd UniMiSS-code/UniMiSS/Downstream/MiTnnu/run

* Run `MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='0' -network='3d_fullres' \
-network_trainer='TrainerV2_BCV' -task='17' -outpath='UniMiss' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=100 \
-pre_train -pre_path='/path/UniMiss_small.pth'` for training.

### 3. Validation 
* Run `MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='0' -network='3d_fullres' \
-network_trainer='TrainerV2_BCV' -task='17' -outpath='UniMiss' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=100 \
-pre_train -pre_path='/path/UniMiss_small.pth' -val --val_folder='validation_output` for validation.

### 4. Acknowledgements
Part of codes are reused from the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet). Thanks to Fabian Isensee for the codes of nnU-Net.

### Contact
Yutong Xie (yutong.xie678@gmail.com)
