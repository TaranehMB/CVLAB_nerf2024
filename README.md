# Visualizing NeRFs

The codes within this directory are for visualizing NeRFs only via the wights of the neural network representing them. The main goal of the code is to recieve weights of a NeRF representation of a specific 3D object and produce a multi-view representation of the object from several angles, outputed as a video. The steps required to do so consist of first, setting up the environment necessary to run the codes, installing the dependencies and then, running the code. 

In the following sections, I will provide a step-by-step guide for utilizing this repository effectiely. 

## Setup

The setup necessary for running the codes within this repository, follows the same procedure found in [CVlab official **nf2vec** repository](https://github.com/CVLAB-Unibo/nf2vec).

### Machine Configuration

The configuration utilized for this code is as follows:
- python==3.8.18
- torch==1.12.0+cu113
- torchvision==0.13.0+cu113
- nerfacc==0.3.5 (with the proper CUDA version set)
- wandb==0.16.0

### Setting Up the Virtual Environment

First, create a virtual environment for installing necessary libraries and dependancies. This procedure can be directly done via python's API, following the procedure below:
```
$ python3 -m venv .venv
$ source .venv/bin/activate
```

Or alternatively, in case you want to need your virtual environment operate a specific version of python compatible with the following libraries and dependancies, you can make the virtual environment via conda (by either having conda or miniconda installed) as follows:
```
$ conda create -n env-name python=3.8.18
```
### Installing Dependencies

By following the commands step-by-step, the environment necessary for excecuting the codes within this repository. It is a given that you already have your virtual environment with python 3.8.18 and pip ready to go. 

1. Install PyTorch and Torchvision:
```
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113`
```
2. Install CUDA Toolkit:
```
conda install -c "nvidia/label/cuda-11.7.1" cuda-toolkit
```
3. Install Ninja and Tiny CUDA NN:
```
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```
4. Install NerfAcc:
```
pip install nerfacc==0.3.5 -f https://nerfacc-bucket.s3.us-west-2.amazonaws.com/whl/torch-1.12.0_cu113.html
```
5. Install Einops:
```
conda install -c conda-forge einops
```
6. Install ImageIO:
```
conda install -c conda-forge imageio
```
7. Install h5py:
```
conda install -c anaconda h5py
```
8. Install TorchMetrics:
```
pip install torchmetrics
```
## Visualizing NeRFs
The code provided here for visualizing NeRFs, called `nerf_viz` is capable of visualizing the dataset provided within `shapenet_render` folder available on the repository. The two other utility codes are necessary for running the code to get the video output. 

The code is capable of generating novel views of the object contained in each seperate folder, assigend to a different shape, doing a 360 degrees camera turn around the object and concatenating these frames to produce a video, producing both a `.gif` and `.mp4` output. 

The results of the code applied to each NeRF is already present within each folder. 
