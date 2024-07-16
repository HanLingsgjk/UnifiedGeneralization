#  UniOS: Unifying Self-supervised Generalization Framework for Optical Flow and Stereo in Real-world

This code contains the content of the paper: ADFactory: An Effective Framework for Generalizing Optical Flow with Nerf (CVPR2024)


https://github.com/user-attachments/assets/3c9e68e2-753c-4507-9934-89f5f7c7a8e6


https://github.com/user-attachments/assets/57c49ebb-120b-4f87-809f-17090cee41c3



https://github.com/user-attachments/assets/2332bc5a-fc5e-45da-9fd1-e39b0b91e515




# Installation
```
conda create -y -n Unios python=3.8
conda activate Unios 

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```
If you don't have CUDAtoolkits, you need to install them yourself, and the version corresponds to PyTorch

Recommended installation method on Nvidia official website  https://developer.nvidia.com/cuda-toolkit-archive
```
conda install cudatoolkit-dev=11.x -c conda-forge    11.x为对应的CUDAtoolkits版本
```

```
pip install setuptools==69.5.1
pip install imageio
pip install scikit-image
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/
```

Install Segment Anything
```
pip install git+https://github.com/facebookresearch/segment-anything.git
```
Download SAM weights and place them in the corresponding path

# Running 

In this section, we introduce the steps required to run the code.

Taking the commonly used 360-v2 as an example, the first step is to download the dataset.
## Dataset
Please download the data from the [Mip-NeRF 360](https://jonbarron.info/mipnerf360/)


Data_path -> Remember to replace the location where the data is stored with your own path

For example:  /media/lh/extradata/Nerf_sence/0280

3d_path -> Remember to replace the location for storing 3D reconstruction data with your own path

For example:  Nerf_sence/0280 (This is a relative path, usually placed in the project folder)

### 1. Reconstructing scenes based on photo groups
```
CUDA_VISIBLE_DEVICES=0 python train.py -s Data_path -m 3d_path -r 4 --port 6312 --kernel_size 0.1
```
### 2. Generate stereo dataset based on reconstruction of scenes
```
 CUDA_VISIBLE_DEVICES=0 python render_stereo.py -m 3d_path --data_device cpu
```
### 3. Generate stereo 3D flight prospects based on reconstructed scenes
```
 CUDA_VISIBLE_DEVICES=0 python render_fore3D_Stereo.py -m 3d_path --data_device cpu
```

### 4. Generate optical flow dataset based on reconstruction of scenes
```
 CUDA_VISIBLE_DEVICES=0 python render_flow.py -m 3d_path --data_device cpu
```
### 5. Generate optical flow 3D flight prospects based on reconstructed scenes
```
 CUDA_VISIBLE_DEVICES=0 python render_fore3D_flow.py -m 3d_path --data_device cpu
```
