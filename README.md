# Automatic-Detection-And-Tracking
Objects detection in the first frame and Tracking special object by SiamRPN.
## Overview
This repo illustrates a automatic detection and tracking of single object. In the process, It first detects all the objects in the first frame of input videos. Next, we should input a examplar image and it can determine the initial position of the target that is most similar to the examplar image. Finally the tracker could finish the single object tracking.
## Key files in this repo
  * demo.py -- implements the detection, identify and tracking pipeline.
  * `detection folder` -- Faster RCNN detection
  * `identify folder` -- phash to identify tracking object
  * `videos folder` -- videos needed to handle
  * examplar.png -- a snapshot of object to track
## Notes
### Detection
It use Faster RCNN to finish object detection. This code was based on longcw's repo [longcw/faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch). It will be improved according to the latest papers.
### Identify
It use **phash** to identify a special object.I will add Siamese Net and traditional Digital image processing to do it in the future.
### Tracking
It use SiamRPN to finish object tracking. The codes was based on [huanglianghua/siamrpn-pytorch](https://github.com/huanglianghua/siamrpn-pytorch). It will be improved according to the latest papers(**DSiamRPN**).
### Prerequisites
* Python 3.6
* PyTorch 0.4.0 or higher
* CUDA 8.0 or higher
### For example
In **Detection** stage. It will detects all cars in the first frame as shown below.
![image](https://github.com/mj000001/Object-Detection-And-Tracking/blob/master/others/out.jpg)

In **Identify** stage. We want to track the car as shown below. It could determine the initial position of the target based on **Detection** stage.  
![image](https://github.com/mj000001/Object-Detection-And-Tracking/blob/master/examplar.png)

In **Tracking** stage. It will track the car.
## Installation and demo
  1. Clone the code:
  ```
  git clone https://github.com/mj000001/Object-Detection-And-Tracking.git
  ```
  2. Create a folder:
  ```
  cd Object-Detection-And-Tracking && mkdir pretrained
  ```
  3. Pretrained Model:
  
  In the root directory of `Object-Detection-And-Tracking`:Download the pretrained `model.pth` and `VGGnet_fast_rcnn_iter_70000`  from [Baidu Yun](https://pan.baidu.com/s/1hoTAVaREj4oZrc8HdtNlDA) with extraction code **gm4f** and put the files under `pretrained/`.  
  
  4. Compilation:
  
  Install python package
  ```
  pip install -r requirements.txt
  ```
  
  Build the Cython modules for nms and the roi_pooling layer
  ```
  cd detection/faster_rcnn
  ./make.sh
  ```
  5. Run Demo:
  ```
  python demo.py
  ```
