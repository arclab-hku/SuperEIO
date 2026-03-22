#ifndef POINT_MATCHING_H_
#define POINT_MATCHING_H_

#include "super_eventmatch.h"
#include "read_configs.h"

class PointMatching{
public:
  PointMatching(Super_EventMatchConfig& super_eventmatch_config);
  int MatchingPoints(const Eigen::Matrix<double, 259, Eigen::Dynamic>& features0, 
      const Eigen::Matrix<double, 259, Eigen::Dynamic>& features1, std::vector<cv::DMatch>& matches,  bool outlier_rejection=true);
  Eigen::Matrix<double, 259, Eigen::Dynamic> NormalizeKeypoints(
      const Eigen::Matrix<double, 259, Eigen::Dynamic> &features, int width, int height);

private:
  Super_EventMatch super_eventmatch;
  Super_EventMatchConfig _super_eventmatch_config;
};

typedef std::shared_ptr<PointMatching> PointMatchingPtr;

#endif  // POINT_MATCHING_H_ 
