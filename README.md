Install
```
conda create -y -n Unios python=3.8
conda activate Unios 

pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118

#如果你没有CUDAtoolkits需要自行安装，版本与pytorch对应
conda install cudatoolkit-dev=11.x -c conda-forge    11.x为对应的CUDAtoolkits版本
#推荐用Nvidia官网的方法装

pip install setuptools==69.5.1 #very important
pip install imageio
pip install scikit-image
pip install -r requirements.txt
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/


#安装Segment Anything
pip install git+https://github.com/facebookresearch/segment-anything.git
下载SAM权重，并放在对应路径
```


运行方式

首先以训练场景 
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
