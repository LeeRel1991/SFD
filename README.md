# S³FD: Single Shot Scale-invariant Face Detector

By [Shifeng Zhang](http://www.cbsr.ia.ac.cn/users/sfzhang/)

### Introduction

S³FD is a real-time face detector, which performs superiorly on various scales of faces with a single deep neural network, especially for small faces. For more details, please refer to our [arXiv paper](https://arxiv.org/abs/1708.05237).

## Features

* real-time
* cxx version can be seen in project [SFDCXX](http://gitlab.bmi/r.li/SFDCXX)

### Install
```shell
# 使用python虚拟环境
sudo apt-get install python-virtualenv
virtualenv -p /usr/bin/python3 py3venv
source py3venv/bin/active


# 安装opencv3.3.0-python
# 参考bmi安装opencv3过程：
# http://gitlab.bmi/VisionAI/soft/Dockerfiles/blob/master/opencv.Dockerfile

# 安装bmi-caffe
# 参考caffe bmi-beta分支的安装过程：
# http://gitlab.bmi/VisionAI/soft/caffe/tree/bmi-beta



# 克隆工程，安装python依赖库
git clone -b bmi http://gitlab.bmi/r.li/SFD.git SFD
cd SFD
pip install -r requirements.txt -i  https://pypi.tuna.tsinghua.edu.cn/simple

```

__注：requirements.txt 未测试__

### Usage

* 运行demo

```
source py3venv/bin/active
export PYTHONPATH=$CAFFE/python:$PYTHONPATH
python demo.py videofile.mp4

# 处理图片
python demo.py $images_folder
```

* 引入其他工程

```python
import sys
sys.path.append("./core")
from sfd import SFD

# 人脸检测
sfd = SFD(model_def="./data/SFD_deploy.prototxt",
          model_weights="./data/SFD_weights.caffemodel",
          img_max_side=480.0,
          conf_thresh=0.8,
          gpu_id=0)

# 准备rgb图片

# 检测人脸
dets = sfd.detect(rgb_img)

for line in dets:
    x1, y1, x2, y2, score = line
    print("bbox ", x1, y1, x2, y2)


```
更多请参考`demo.py`

* combination with track

参考'sfd_api.py'

### Eval

【todo】
see `eval/README.md`
