#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "tic_toc.h"

#include "dvs_msgs/Event.h"
#include "dvs_msgs/EventArray.h"
#include "TS_generator/TS_generator.h"
#include "utility/visualization.h"

#include "read_configs.h"
#include "super_eventpoint.h"
#include "global_config.h"
#include <fstream>
#include <iomanip>


using namespace std;
using namespace camodocal;
using namespace Eigen;


bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker();

    void build_super_eventpoint(Configs& configs);

    void readEvent(const dvs_msgs::EventArray &last_event, double _cur_time);

    void setevent_Mask();

    void addPoints();

    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void rejectWithF_event();

    cv::Mat getTrackImage();
    cv::Mat getTrackImage_two();
    cv::Mat gettimesurface();

    double distance(cv::Point2f &pt1, cv::Point2f &pt2);

    void event_drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                                   vector<int> &curLeftIds,
                                   vector<cv::Point2f> &curLeftPts, 
                                   vector<cv::Point2f> &curRightPts,
                                   map<int, cv::Point2f> &prevLeftPtsMap);
    
    void event_drawTrack_two(const cv::Mat &imLeft, const cv::Mat &imRight, 
                                vector<int> &curLeftIds,
                                vector<cv::Point2f> &curLeftPts, 
                                vector<cv::Point2f> &curRightPts,
                                map<int, cv::Point2f> &prevLeftPtsMap);

    vector<cv::Point2f> undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam);

    vector<cv::Point2f> ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                    map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts);
    cv::Mat mask;
    cv::Mat mask_arc;
    cv::Mat fisheye_mask;
    cv::Mat prev_img, cur_img, forw_img;
    vector<cv::Point2f> n_pts;
    vector<cv::Point2f> prev_pts, cur_pts, forw_pts;
    vector<cv::Point2f> prev_un_pts, cur_un_pts;
    vector<cv::Point2f> pts_velocity;
    vector<int> ids;
    vector<int> track_cnt;
    map<int, cv::Point2f> cur_un_pts_map;
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr m_camera;
    double cur_time;
    double prev_time;

    Eigen::Matrix<double, 259, Eigen::Dynamic> points0;
    Eigen::Matrix<double, 259, Eigen::Dynamic> points1;

    std::vector<cv::DMatch> matches;

    std::mutex _gpu_mutex;


    cv::Mat imTrack;
    cv::Mat imTrack_two;
    cv::Mat time_surface_visualization;
    cv::Mat time_surface_without_polarity_visualization;

    map<int, cv::Point2f> prevLeftPtsMap;
    vector<cv::Point2f> cur_right_pts;

    bool FLAG_DETECTOR_NOSTART=true;

    static int n_id;

  private:
    std::shared_ptr<Super_EventPoint> super_eventpoint;

};
