#  UniOS: Unifying Self-supervised Generalization Framework for Optical Flow and Stereo in Real-world

# Installation
```
conda create -y -n Unios python=3.8
conda activate Unios 

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```
If you don't have CUDAtoolkits, you need to install them yourself, and the version corresponds to PyTorch
```
conda install cudatoolkit-dev=11.x -c conda-forge    11.x为对应的CUDAtoolkits版本
```
Recommended installation method on Nvidia official website
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

## Running

In this section, we introduce the steps required to run the code.

Taking the commonly used 360-v2 as an example, the first step is to download the dataset.

```
CUDA_VISIBLE_DEVICES=0 python train.py -s /media/lh/extradata/Nerf_sence/0280 -m NerF_sence/0280 -r 4 --port 6312 --kernel_size 0.1
```
#生成双目立体匹配数据集
```
 CUDA_VISIBLE_DEVICES=0 python render_stereo.py -m NerF_sence/0280 --data_device cpu
```
#生成用于双目立体匹配的3D飞行前景
```
 CUDA_VISIBLE_DEVICES=0 python render_fore3D_Stereo.py -m NerF_sence/0280--data_device cpu
```

#生成光流数据集
```
 CUDA_VISIBLE_DEVICES=0 python render_flow.py -m NerF_sence/0280 --data_device cpu
```
#生成用于光流的3D飞行前景
```
 CUDA_VISIBLE_DEVICES=0 python render_fore3D_flow.py -m NerF_sence/0280 --data_device cpu
```
