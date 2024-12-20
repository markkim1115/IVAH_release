# IVAH_release
Official PyTorch implementation of "IVAH: Invisible-Region Augmented Human Rendering from Monocular Images"

## H/W Requirements
IVAH is trained and tested on an RTX A6000 GPU, we recommend to use GPU that has more than 24GB GPU memory to train the model.

## Install
This framework is tested on Ubuntu 20.04, Python3.8 and CUDA 11.3

### 1. Install Pytorch 1.11.0
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
or
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

### 2. Install Pytorch3D 0.7.2
https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md

```
pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
```

### 3. Install requirements.txt
```
pip install -r requirements.txt
```

### 4. Download pre-trained weights of BackNet and IVAH

Download pre-trained weights from below link:
https://www.dropbox.com/scl/fo/gzthflt7klqzjtiaxnvgn/AHAXkJSsEG0l57uiEPLT3d8?rlkey=mch033t1qsqobwlcry33btz6t&st=2su67umf&dl=0

### 5. Put downloaded weights in correct paths

* Put downloaded BackNet weights in back_generator_ckpts like below:
```
root
  └── back_generator_ckpts
      ├── humman
      │   ├── val_results_humman.txt
      │   └── weights
      │       ├── GNR-model-snapshot-0080.txt
      │       └── humman_backnet_060.pth
      ├── renderpeople
      │   ├── val_results_RenderPeople.txt
      │   └── weights
      │       └── renderpeople_backnet_040.pth
      ├── thuman1
      │   ├── val_results_thuman1.txt
      │   └── weights
      │       └── thuman1_backnet_060.pth
      └── thuman2
          ├── val_results_thuman2.txt
          └── weights
              └── thuman2_backnet_060.pth
```
* Put downloaded trained IVAH checkpoints into the experiments directory like below:
```
root
  └── experiments
      ├── humman
      │   └── full_model
      │       ├── <yaml file>
      │       └── <ckpt file(.tar)>
      ├── renderpeople
      │   ├── full_model
      │       ├── <yaml file>
      │       └── <ckpt file(.tar)>
      ├── thuman1
      │   └── full_model
      │       ├── <yaml file>
      │       └── <ckpt file(.tar)>
      └── thuman2
          └── full_model
              ├── <yaml file>
              └── <ckpt file(.tar)>
```