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
