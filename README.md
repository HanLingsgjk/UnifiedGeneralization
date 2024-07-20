#  UniOS: Unifying Self-supervised Generalization Framework for Optical Flow and Stereo in Real-world

This code contains the content of the paper: ADFactory: An Effective Framework for Generalizing Optical Flow with Nerf (CVPR2024)

This project introduces a self-supervised generalization method for training optical flow and stereo tasks, which can generate high-quality optical flow and stereo datasets by simply inputting RGB images captured by the camera



https://github.com/user-attachments/assets/67b1ba10-8db4-4425-822f-6372830f2caa



https://github.com/user-attachments/assets/8219cffb-ea6b-427d-9cda-b673d5db57fd



https://github.com/user-attachments/assets/54ecef75-c398-4796-86c6-701973204f23




![stereog](https://github.com/user-attachments/assets/e7849b92-57d4-444d-8eb9-783682def986)



# Installation
## Install mip-splatting

please note that we have made modifications to the rasterization code here (for rendering radiation field confidence and median depth), which is different from the original version
```
conda create -y -n Unios python=3.8
conda activate Unios 

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```
If you don't have CUDAtoolkits, you need to install them yourself, and the version corresponds to PyTorch

Recommended installation method on Nvidia official website  https://developer.nvidia.com/cuda-toolkit-archive
11. x is the corresponding CUDA toolkit version
```
conda install cudatoolkit-dev=11.x -c conda-forge  
```
Install key packages and compile rasterized code
```
pip install setuptools==69.5.1
pip install imageio
pip install scikit-image
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/
```

## Install Segment Anything
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
Download SAM weights and place them in the corresponding path
https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
The paths that need to be changed are:
1. In   render_fore3D_flow.py  line36
```
 sam = sam_model_registry['vit_h'](checkpoint='/home/lh/Track-Anything/checkpoints/sam_vit_h_4b8939.pth')
```
2. In   render_fore3D_Stereo.py  line36
```
 sam = sam_model_registry['vit_h'](checkpoint='/home/lh/Track-Anything/checkpoints/sam_vit_h_4b8939.pth')
```
# Running 

In this section, we introduce the steps required to run the code.

Taking the commonly used 360-v2 as an example, the first step is to download the dataset.
## Dataset
Please download the data from the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/) Extract the data and place it in your preferred location, such as /media/lh/


Data_path -> Remember to replace the location where the data is stored with your own path

For example:  /media/lh/extradata/360_v2/kitchen

3d_path -> Remember to replace the location for storing 3D reconstruction data with your own path

For example:  3d_sence/kitchen (This is a relative path, usually placed in the project folder)

### 1. Reconstructing scenes based on photo groups
```
CUDA_VISIBLE_DEVICES=0 python train.py -s Data_path -m 3d_path -r 4 --port 6312 --kernel_size 0.1
```
## Generate dataset
The storage path for generating the dataset needs to be manually set, for example, in render_stereo.py it is in line243
```
dataroot = '/home/lh/all_datasets/MIPGS10K_stereotest'
```
in render_fore3D_Stereo.py it is in line257

in render_flow.py it is in line263

in render_fore3D_flow.py it is in line305


### 2. Generate stereo dataset based on reconstruction of scenes
```
 CUDA_VISIBLE_DEVICES=0 python render_stereo.py -m 3d_path --data_device cpu
```
### 3. Generate stereo 3D flight foregrounds based on reconstructed scenes
```
 CUDA_VISIBLE_DEVICES=0 python render_fore3D_Stereo.py -m 3d_path --data_device cpu
```

### 4. Generate optical flow dataset based on reconstruction of scenes
```
 CUDA_VISIBLE_DEVICES=0 python render_flow.py -m 3d_path --data_device cpu
```
### 5. Generate optical flow 3D flight foregrounds based on reconstructed scenes
```
 CUDA_VISIBLE_DEVICES=0 python render_fore3D_flow.py -m 3d_path --data_device cpu
```
