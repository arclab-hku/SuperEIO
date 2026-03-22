#include "point_matching.h"

#include <opencv2/opencv.hpp>

PointMatching::PointMatching(Super_EventMatchConfig& super_eventmatch_config) :super_eventmatch(super_eventmatch_config){
  _super_eventmatch_config = super_eventmatch_config;
  if (!super_eventmatch.build()){
    std::cout << "Error in super_eventmatch building" << std::endl;
    exit(0);
  }
}

int PointMatching::MatchingPoints(const Eigen::Matrix<double, 259, Eigen::Dynamic>& features0, 
    const Eigen::Matrix<double, 259, Eigen::Dynamic>& features1, std::vector<cv::DMatch>& matches, bool outlier_rejection){
  matches.clear();
  Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features0 = NormalizeKeypoints(features0, _super_eventmatch_config.image_width, _super_eventmatch_config.image_height);
  Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features1 = NormalizeKeypoints(features1, _super_eventmatch_config.image_width, _super_eventmatch_config.image_height);
  Eigen::VectorXi indices0, indices1;
  Eigen::VectorXd mscores0, mscores1;
  super_eventmatch.infer(norm_features0, norm_features1, indices0, indices1, mscores0, mscores1);

  int num_match = 0;
  std::vector<cv::Point2f> points0_out, points1_out;
  std::vector<int> point_indexes;
  for(size_t i = 0; i < indices0.size(); i++){
    if(indices0(i) < indices1.size() && indices0(i) >= 0 && indices1(indices0(i)) == i){
      double d = 1.0 - (mscores0[i] + mscores1[indices0[i]]) / 2.0;
      matches.emplace_back(i, indices0[i], d);
      points0_out.emplace_back(features0(1, i), features0(2, i));
      points1_out.emplace_back(features1(1, indices0(i)), features1(2, indices0(i)));
      num_match++;
    }
  }

  //输出匹配的对数
  // std::cout << "ori num_match: " << matches.size() << std::endl;

  // reject outliers
  if(outlier_rejection){
    std::vector<uchar> inliers;
    if (points1_out.size() >= 8){
      cv::findFundamentalMat(points0_out, points1_out, cv::FM_RANSAC, 3, 0.99, inliers);
      int j = 0;
      for(int i = 0; i < matches.size(); i++){
        if(inliers[i]){
          matches[j++] = matches[i];
        }
      }
      matches.resize(j);
    }
  }

  //输出匹配的对数
  // std::cout << "RANSAC num_match: " << matches.size() << std::endl;

  return matches.size();
}

Eigen::Matrix<double, 259, Eigen::Dynamic> PointMatching::NormalizeKeypoints(const Eigen::Matrix<double, 259, Eigen::Dynamic> &features,
                         int width, int height) {
  Eigen::Matrix<double, 259, Eigen::Dynamic> norm_features;
  norm_features.resize(259, features.cols());
  norm_features = features;
  for (int col = 0; col < features.cols(); ++col) {
    norm_features(1, col) =
        (features(1, col) - width / 2) / (std::max(width, height) * 0.7);
    norm_features(2, col) =
        (features(2, col) - height / 2) / (std::max(width, height) * 0.7);
  }
  return norm_features;
}
