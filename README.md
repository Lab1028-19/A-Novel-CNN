## A Novel CNN architecture for Real-time Point Cloud Feature Recognition in Road Environment

This repository contains a contains two models, which are used for road semantics segmentation and moving object semantics segmentation. They all use the following structure of neural network.

<p align="center">
    <img src="https://github.com/Lab1028-19/A-Novel-CNN/blob/master/img/Multichannel%20point%20feature%20map%20convolutional%20neural%20network%20for%20point%20cloud.png" width="600" />
</p>


### Introduction
This model focuses on the semantic segmentation task of point cloud recognition in the road environment and proposes a new method for detecting obstacles and semantic segmentation in LIDAR in an unmanned environment. Through the study of the characteristics of road environment point cloud feature map and the special module ablation comparison analysis experiment, it is proved the validity and applicability of the related structure design of the point cloud semantic segmentation model.Experiments show that the proposed method has good accuracy and real-time performance in the public point cloud datasets Apollo and KITTI .


## Installation:

The instructions are tested on Ubuntu 16.04 with python 3.5(Anaconda)and tensorflow 1.14 with GPU support.

- Clone the A-Novel-CNN repository:

    ```Shell
    git clone https://github.com/Lab1028-19/A-Novel-CNN.git
    ```
    
  We name the root directory as `$SQSG_ROOT`.


- Use conda to install tensorflow(Automatically install the latest GPU versions, including CUDA and cudnn):

    ```Shell
    conda install tensorflow
    ```


## Training/Validation
### Road segmentationd
- First,decompression training data.

    ```Shell
    cd $SQSG_ROOT/road_pointcloud/data/
    wget https://cmnet.oss-cn-hongkong.aliyuncs.com/road_pointcloud.tar.gz
    tar -xzvf road_pointcloud.tar.gz
    rm road_pointcloud.tar.gz
    ```

- Now we can start training by
    ```Shell
    cd $SQSG_ROOT/
    python train.py
    ```
   Training logs and model checkpoints will be saved in the log directory.
   
- We can launch evaluation script simutaneously with training

    ```Shell
    cd $SQSG_ROOT/
    python evaluate.py
    ```
    
### Obstacle segmentationd
 - This part of the experimental operation refers to the above.
 - Download training data.

    ```Shell
    cd $SQSG_ROOT/obstacle/data/
    wget https://cmnet.oss-cn-hongkong.aliyuncs.com/obstacle.tar.gz
    tar -xzvf obstacle.tar.gz
    rm obstacle.tar.gz
    ```
## Experiment
<p align="center">
    <img src="https://github.com/Lab1028-19/A-Novel-CNN/blob/master/img/Ablation.PNG" width="600" />
</p>

<p align="center">
    <img src="https://github.com/Lab1028-19/A-Novel-CNN/blob/master/img/semantic.PNG" width="600" />
</p>

<p align="center">
    <img src="https://github.com/Lab1028-19/A-Novel-CNN/blob/master/img/road.PNG" width="600" />
</p>





