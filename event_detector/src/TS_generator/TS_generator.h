#ifndef TS_GENERATOR_H
#define TS_GENERATOR_H

#include <Eigen/Dense>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "../parameters.h"
#include <cmath>
#include <thread>

namespace TS {

class TSGenerator
{
public:
  TSGenerator();
  ~TSGenerator();

void init(int col, int row);
void createSAE(double et, int ex, int ey, bool ep);
cv::Mat SAEtoTimeSurface(const double external_sync_time);

cv::Mat cur_event_mat;


private:

  int kSensorWidth_;
  int kSensorHeight_;

  cv::Size sensor_size_;
  double decay_ms_;
  bool ignore_polarity_;
  int median_blur_kernel_size_;
  double decay_ms_for_loop;
  double filter_threshold_;

  // Surface of Active Events
  Eigen::MatrixXd sae_[2];
  Eigen::MatrixXd sae_latest_[2];

};


}

#endif
