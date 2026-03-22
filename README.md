<div align="center">

# SuperEIO: Self-Supervised Event Feature Learning for Event Inertial Odometry 

**Peiyu Chen**<sup>†</sup>, **Fuling Lin**<sup>†</sup>, **Weipeng Guan**, **Yi Luo**, **Peng Lu**<sup>*</sup>

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

You should modifie your `.bashrc` file through `gedit ~/.bashrc`, add the following codes in it:
~~~
source ~/catkin_ws_supereio/devel/setup.bash
alias supereiobuild='cd /home/YOUR_name/catkin_ws_supereio && catkin build -j8 supereio_ba event_detector loop_closure -DCMAKE_BUILD_TYPE=Debug'
alias supereiorun='cd /home/YOUR_name/catkin_ws_supereio/src/SuperEIO/script && sh run.sh'
~~~

After that, run the `source ~/.bashrc` and `supereiobuild` command in your terminal.

## 3. Run on Dataset

### 3.1 Run on Examples
You can test our SuperEIO on [hku_agg_translation](https://github.com/arclab-hku/Event_based_VO-VIO-SLAM). After you download bag files, just run the example:
~~~
roslaunch supereio_ba hku_stereo.launch
rosbag play YOUR_DOWNLOADED.bag
~~~

or you can just revise the script file and run in your terminal:
~~~
supereiorun
~~~

### 3.2 Run on Your Datasets
To run the system on your dataset, you need to create a corresponding configuration folder and YAML file in the ’config‘ directory. Then configure your camera intrinsics, event/IMU topics, and the extrinsic transformation between the event camera and IMU in the YAML file. For the extrinsic calibration, we recommend you follow the link ([DVS-IMU Calibration and Synchronization](https://arclab-hku.github.io/ecmd/calibration/)) to kindly calibrate your sensors.

Following that, you can execute the provided command to run SuperEIO on your dataset:
~~~
roslaunch supereio_ba YOUR_DATASET.launch
rosbag play YOUR_BAG.bag
~~~


## 4. Detection and Matching Performance
We present the qualitative performance of our event detetor and descriptor matcher on multiple public datasets.

<div align="center">
    <img src="https://github.com/cpymaple/paper_pic/blob/main/supereio/feature_detection.png" width="80%" />
    <p>Visual comparison of other event feature detectors, SuperPoint, and ours on multiple datasets with corresponding images. From top to bottom: DAVIS240C, Mono HKU, Stereo HKU, and VECtor.</p>
</div>

<div align="center">
    <img src="https://github.com/cpymaple/paper_pic/blob/main/supereio/descriptor_matching.png" width="80%" />
    <p>Examples of our event descriptor matches in loop closure under boxes translation and hku agg translation sequences.</p>
</div>



## 5. Video Demo
We present video demo of our SuperEIO system, showcasing its visulization performance on both hdr_boxes and aggressive_flight scenarios.

<div align="center">
    <img src="https://github.com/cpymaple/paper_pic/blob/main/supereio/Supereio240hdrboxes.gif" width="49%" />
    <img src="https://github.com/cpymaple/paper_pic/blob/main/supereio/Supereioaggflight.gif" width="49%" />
    <p>The Video Demo of Our SuperEIO on Hdr_boxes and Aggressive_flight Sequences</p>
</div>


## 6. Citation
<!-- SuperEIO is published in [Journal/Conference Name] with [Presentation Option]. (The [Journal/Conference Name] pdf is available [here](YOUR_PDF_LINK) and the arxiv pdf is available [here](YOUR_ARXIV_LINK)). -->
SuperEIO is available in the [Arxiv](http://arxiv.org/abs/2503.22963).
~~~
@article{SuperEIO,
  title={SuperEIO: Self-Supervised Event Feature Learning for Event Inertial Odometry},
  author={Chen, Peiyu and Lin, Fuling and Guan, Weipeng and Luo, Yi and Lu, Peng},
  journal={IEEE Transactions on Industrial Electronics},
  year={2026}
}
~~~

If you feel like SuperEIO has indeed helped in your current research or work, a simple star or citation of our works should be the best affirmation for us. :blush: 

## 7. Acknowledgments 
This work was supported by the General Research Fund under Grant 17204222, and in part by the Seed Fund for Collaborative Research and General Funding Scheme-HKU-TCL Joint Research Center for Artificial Intelligence. We gratefully acknowledge [sair-lab/AirSLAM](https://github.com/sair-lab/AirSLAM) for providing the Superpoint TensorRT acceleration template, which significantly enhanced the compute effiency of our system.

## 8. License
The source code is released under the GPLv3 license. We are still working on improving the code reliability. If you are interested in our project for commercial purposes, please contact [Dr. Peng LU](https://arclab.hku.hk/People.html) for further communication.
