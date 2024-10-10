#  Self-Assessed Generation: Trustworthy Label Generation for Optical Flow and Stereo Matching in Real-world

This code contains the content of the paper: ADFactory: An Effective Framework for Generalizing Optical Flow with Nerf (CVPR2024)

This project introduces a self-supervised generalization method for training optical flow and stereo tasks, which can generate high-quality optical flow and stereo datasets by simply inputting RGB images captured by the camera

**We provided weights for ScaleFlow++ trained using GS58 in [Project](https://github.com/HanLingsgjk/CSCV) Readers can reproduce the effect in Table V by downloading Demo_Scaleflowpp.pth**

**The complete GS58 dataset consists of 652G background data and 334GB foreground data, and we have yet to find a portable public method. We suggest you generate it yourself. Our data sources mainly include 270 scenes provided by [fabiotosi92/NeRF Superior Deep Stereo: A novel paradigm for collecting and generating stereo training data using neural rendering (GitHub)](https://nerfstereo.github.io/), 9 scenes provided by mip-NeRF 360 [(jonbarron. info)](https://jonbarron.info/mipnerf360/), and some still life shots we took using Pocket3 (toy cars, etc.). Using the first two data sources is already sufficient to achieve most of the promised performance.**

https://github.com/user-attachments/assets/a88bd218-77c4-47c8-ac68-4c2a8cf7ed19



https://github.com/user-attachments/assets/57bd1456-5bba-47b8-a268-22cb22776e4f




https://github.com/user-attachments/assets/1fce5ca0-bfab-4b54-b0f1-ee1e0d8c5441




https://github.com/user-attachments/assets/ca95ecc9-1501-4e01-90ad-38c7a53ccd19




https://github.com/user-attachments/assets/da9af66f-d3fb-4f0b-8705-85bae435559d




https://github.com/user-attachments/assets/8baf1a5c-6e12-418e-b5cb-f4a3fde0746a




https://github.com/user-attachments/assets/58cd07e9-499b-4c5b-84e8-6297e649989b



![git1](https://github.com/user-attachments/assets/3409a9aa-1cf4-4ce3-bd7d-12d5e177278d)


![git2](https://github.com/user-attachments/assets/42b1581b-f68b-45cc-b6c0-c577813cb8c7)



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
