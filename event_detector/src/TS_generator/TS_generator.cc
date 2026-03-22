#include "TS_generator.h"
#include <ros/ros.h>


namespace TS  {

TSGenerator::TSGenerator() {
    ROS_INFO("Generate TS");
}

TSGenerator::~TSGenerator() {
}

void TSGenerator::init(int col, int row){

    kSensorWidth_=col;
    kSensorHeight_=row;
    sensor_size_ = cv::Size(kSensorWidth_, kSensorHeight_);
    sae_[0] = Eigen::MatrixXd::Zero(kSensorWidth_, kSensorHeight_);
    sae_[1] = Eigen::MatrixXd::Zero(kSensorWidth_, kSensorHeight_);
    sae_latest_[0] = Eigen::MatrixXd::Zero(kSensorWidth_, kSensorHeight_);
    sae_latest_[1] = Eigen::MatrixXd::Zero(kSensorWidth_, kSensorHeight_);

    decay_ms_=para_decay_ms;
    ignore_polarity_= (bool) para_ignore_polarity;//true;
    median_blur_kernel_size_=para_median_blur_kernel_size;//1;
    filter_threshold_=para_feature_filter_threshold;

}

void TSGenerator::createSAE(double et, int ex, int ey, bool ep){

      // Update Surface of Active Events
      const int pol = ep ? 1 : 0;
      const int pol_inv = (!ep) ? 1 : 0;

      double & t_last = sae_latest_[pol](ex,ey);
      double & t_last_inv = sae_latest_[pol_inv](ex, ey);

      if ((et > t_last + filter_threshold_) || (t_last_inv > t_last) ) {
        t_last = et;
        sae_[pol](ex, ey) = et;
      } else {
        t_last = et;
      }

      cur_event_mat.at<cv::Vec3b>(cv::Point(ex,ey)) = (
            ep == true ? cv::Vec3b(0, 0, 255) : cv::Vec3b(255, 0, 0));
}

cv::Mat TSGenerator::SAEtoTimeSurface( const double external_sync_time){

    // create exponential-decayed Time Surface map.
    const double decay_sec = decay_ms_ / 1000.0;
    cv::Mat time_surface_map;
    time_surface_map=cv::Mat::zeros(sensor_size_, CV_64F);

      // Loop through all coordinates
      for (int y=0;y<sensor_size_.height;++y){
              for(int x=0;x<sensor_size_.width;++x){

                    double most_recent_stamp_at_coordXY= (sae_[1](x,y)>sae_[0](x,y)) ? sae_[1](x,y) : sae_[0](x,y);

                    if(most_recent_stamp_at_coordXY > 0){
                          const double dt = (external_sync_time-most_recent_stamp_at_coordXY);
                          double expVal =std::exp(-dt / decay_sec);

                          if(!ignore_polarity_)
                            {
                                double polarity = (sae_[1](x,y)>sae_[0](x,y)) ? 1.0 : -1.0;  

                                expVal *= polarity;
                            }
                            time_surface_map.at<double>(y,x) = expVal;
                    }
              }
      }

      // polarity
      if(!ignore_polarity_)
        time_surface_map = 255.0 * (time_surface_map + 1.0) / 2.0;
      else
        time_surface_map = 255.0 * time_surface_map;
      time_surface_map.convertTo(time_surface_map, CV_8U);

      // median blur
      if(median_blur_kernel_size_ > 0){
          cv::medianBlur(time_surface_map, time_surface_map, 2 * median_blur_kernel_size_ + 1);
      }

    return time_surface_map;
}

}
