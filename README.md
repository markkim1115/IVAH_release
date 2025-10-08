# IVAH: Invisible-Region Augmented Human Rendering from Monocular Images
Official PyTorch implementation of "IVAH: Invisible-Region Augmented Human Rendering from Monocular Images (IEEE Access, 2025)"
[[PDF]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=11192415)
<!-- ![itw](assets/itw_example.png) -->

<p align="center">
    <img src="assets/itw_example.jpg" width="700px"/>
</p>

## Hardware Requirements
IVAH is trained and tested on an RTX A6000 GPU. We recommend using a GPU with more than 24GB of memory to train the model.

## Installation
This framework is tested on Ubuntu 20.04, Python 3.8, and CUDA 11.3.

### 1. Install PyTorch 1.11.0
```
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```
or
```
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

### 2. Install psbody-mesh and PyTorch3D
(1) Install psbody-mesh from this [repo](https://github.com/MarilynKeller/mesh). This repository provides a modified version of the official library to resolve dependency issues.

(2) Install PyTorch3D using the command below or from the [official page](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md):

```
pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1110/download.html
```

### 3. Install dependencies
```
pip install -r requirements.txt
```

### 4. Download pre-trained weights of BackNet and IVAH

Download pre-trained weights from the following link:
[Download Link](https://www.dropbox.com/scl/fo/gzthflt7klqzjtiaxnvgn/AHAXkJSsEG0l57uiEPLT3d8?rlkey=mch033t1qsqobwlcry33btz6t&st=2su67umf&dl=0)

### 5. Place downloaded weights in the correct paths

* Place the downloaded BackNet weights in the `back_generator_ckpts` directory as shown below:
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
* Place the downloaded trained IVAH checkpoints into the `experiments` directory as shown below:
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

### 6. Download SMPL model and place in the correct path

(1) Download SMPL from the [SMPL official page](https://smpl.is.tue.mpg.de/). The SMPL model must be placed as described below.
(2) Download the model directory and contents from [here](https://www.dropbox.com/scl/fo/gzthflt7klqzjtiaxnvgn/AHAXkJSsEG0l57uiEPLT3d8?rlkey=mch033t1qsqobwlcry33btz6t&st=2su67umf&dl=0), and place them in the `smpl` directory.

The final directory structure should be as follows:
```
root
  └── third_parties
      └── smpl
          └── models
              ├── PUT_SMPL_MODEL_HERE
              ├── SMPL_NEUTRAL.pkl
              └── UV_data
                   ├── barycentric_h0256_w0256.pkl
                   ├── smpl_uv.obj
                   ├── smpl_uv_20200910.png
                   ├── smpl_uv_20200910_body.png
                   ├── smpl_uv_20200910_face.png
                   └── smpl_uv_20200910_hand.png
```

Now you are ready to run the demo and training code.

## Run Demo

```
sh test_in_the_wild_freeview.sh
```

## Run Training
```
sh train.sh
```
