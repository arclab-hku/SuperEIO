<div align="center">

# SuperEIO: Self-Supervised Event Feature Learning for Event Inertial Odometry 

**Peiyu Chen**<sup>†</sup>, **Fuling Lin**<sup>†</sup>, **Weipeng Guan**, **Peng Lu**<sup>*</sup>

**Adaptive Robotic Controls Lab (ArcLab)**, **The University of Hong Kong**.

</div>


<div align="center">
    <a href="YOUR_PROJECT_WEBSITE_LINK" target="_blank"><img src="https://github.com/cpymaple/paper_pic/blob/main/supereio/framework.png" width="80%" /></a>
    <p>System Overview of SuperEIO</p>
</div>

<p align="justify">
  <strong>SuperEIO</strong> is a novel event-based visual inertial odometry framework that leverages self-supervised learning networks to enhance the accuracy and robustness of ego-motion estimation. Our event-only feature detection employs a convolutional neural network under continuous event streams. Moreover, our system adopts the graph neural network to achieve event descriptor matching for loop closure. The proposed system utilizes TensorRT to accelerate the inference speed of deep networks, which ensures low-latency processing and robust real-time operation on resource-limited platforms. Besides, we evaluate our method extensively on multiple challenging public datasets, particularly in high-speed motion and high-dynamic-range scenarios, demonstrating its superior accuracy and robustness compared to other state-of-the-art event-based methods.
</p>



## 1. Prerequisites
We test our SuperEIO on Ubuntu 20.04. Before you build the SuperEIO, you should install the following dependency:
* Ceres 1.14.0
* OpenCV 4.2
* Eigen 3
* TensorRT 8.4.1.5
* CUDA 11.6
* ROS noetic

Other event camera drivers are stored in the folder `dependencies`.

## 2. Build
~~~
mkdir -p catkin_ws_supereio/src
cd catkin_ws_supereio
catkin config --init --mkdirs --extend /opt/ros/noetic --merge-devel --cmake-args -DCMAKE_BUILD_TYPE=Release
cd ~/catkin_ws_supereio/src
git clone git@github.com:your-repo/SuperEIO.git --recursive
~~~


After that, run the `source ~/.bashrc` and `supereiobuild` command in your terminal.

## 3. Run on Dataset

### 3.1 Run on Examples
You can test our SuperEIO on [hku_agg_translation](https://github.com/arclab-hku/Event_based_VO-VIO-SLAM). After you download bag files, just run the example:
~~~
roslaunch supereio_estimator supereio.launch 
rosbag play YOUR_DOWNLOADED.bag
~~~

### 3.2 Run on Your Datasets
To run the system on your dataset, you need to create a corresponding configuration folder and YAML file in the ’config‘ directory. Then configure your camera intrinsics, event/IMU topics, and the extrinsic transformation between the event camera and IMU in the YAML file. For the extrinsic calibration, we recommend you follow the link ([DVS-IMU Calibration and Synchronization](https://arclab-hku.github.io/ecmd/calibration/)) to kindly calibrate your sensors.

Following that, you can execute the provided command to run SuperEIO on your dataset:
~~~
roslaunch supereio_estimator supereio.launch
rosbag play YOUR_BAG.bag
~~~


## 4. Architecture and Feature Performance
We present the network architectures of our deep event feature detector and descriptor matcher, along with visualizations demonstrating their performance in event feature detection and descriptor matching.

<div align="center">
    <img src="https://github.com/cpymaple/paper_pic/blob/main/supereio/super_eventpoint.png" width="49%" />
    <img src="https://github.com/cpymaple/paper_pic/blob/main/supereio/feature_detection.png" width="49%" />
    <p>The Architecture of Our Event Feature Detector and Comparison with Other Event-based Detector</p>
</div>

<div align="center">
    <img src="https://github.com/cpymaple/paper_pic/blob/main/supereio/super_eventglue.png" width="49%" />
    <img src="https://github.com/cpymaple/paper_pic/blob/main/supereio/descriptor_matching.png" width="49%" />
    <p>The Architecture of Our Event Descriptor Matcher and Loop Closure Performance</p>
</div>


## 5. Citation
<!-- SuperEIO is published in [Journal/Conference Name] with [Presentation Option]. (The [Journal/Conference Name] pdf is available [here](YOUR_PDF_LINK) and the arxiv pdf is available [here](YOUR_ARXIV_LINK)). -->
SuperEIO is available in the [Arxiv](http://arxiv.org/abs/2503.22963).
~~~
@article{SuperEIO,
  title={SuperEIO: Self-Supervised Event Feature Learning for Event Inertial Odometry},
  author={Chen, Peiyu and Lin, Fuling and Guan, Weipeng and Lu, Peng},
  journal={arXiv preprint arXiv:2503.22963},
  year={2025}
}
~~~

If you feel like SuperEIO has indeed helped in your current research or work, a simple star or citation of our works should be the best affirmation for us. :blush: 

## 6. Coming Soon
The full codebase will be released upon paper acceptance. For immediate inquiries, please contact the authors.

## 7. Acknowledgments 
This work was supported by the General Research Fund under Grant 17204222, and in part by the Seed Fund for Collaborative Research and General Funding Scheme-HKU-TCL Joint Research Center for Artificial Intelligence. We gratefully acknowledge [sair-lab/AirSLAM](https://github.com/sair-lab/AirSLAM) for providing the Superpoint TensorRT acceleration template, which significantly enhanced the compute effiency of our system.

## 8. License
The source code is released under the GPLv3 license. We are still working on improving the code reliability. If you are interested in our project for commercial purposes, please contact [Dr. Peng LU](https://arclab.hku.hk/People.html) for further communication.
