#include "keyframe.h"
#include "../../event_detector/src/dvs_msgs/Event.h"
#include "../../event_detector/src/dvs_msgs/EventArray.h"

double total_t_match_super = 0.0;
int num_t_match_super = 0;

template <typename Derived>


static void reduceVector(vector<Derived> &v, vector<uchar> status)// delete the points with status 0
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

KeyFrame::KeyFrame(double _time_stamp, int _index, Vector3d &_vio_T_w_i, Matrix3d &_vio_R_w_i, cv::Mat &_image,
		           vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_norm,
		           vector<double> &_point_id, int _sequence, std::shared_ptr<Super_EventPoint> super_eventpoint)
{ 
	time_stamp = _time_stamp;
	index = _index;
	vio_T_w_i = _vio_T_w_i;
	vio_R_w_i = _vio_R_w_i;
	//record original Pose and loop closure Pose
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
	origin_vio_T = vio_T_w_i;		
	origin_vio_R = vio_R_w_i;
	image = _image.clone();
	cv::resize(image, thumbnail, cv::Size(80, 60));
	point_3d = _point_3d;
	point_2d_uv = _point_2d_uv;
	point_2d_norm = _point_2d_norm;
	point_id = _point_id;
	has_loop = false;
	loop_index = -1;
	has_fast_point = false;
	loop_info << 0, 0, 0, 0, 0, 0, 0, 0;
	sequence = _sequence;
	computeWindowBRIEFPoint_glue(super_eventpoint);
	computeBRIEFPoint_glue(super_eventpoint);

	if(!DEBUG_IMAGE)
		image.release();
}

void KeyFrame::computeWindowBRIEFPoint_glue(std::shared_ptr<Super_EventPoint> super_eventpoint)
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());

	for(int i = 0; i < (int)point_2d_uv.size(); i++)
	{
	    cv::KeyPoint key;
	    key.pt = point_2d_uv[i];
	    window_keypoints.push_back(key);
	}
	extractor(image, window_keypoints, window_brief_descriptors);
	
	Eigen::Matrix<double, 259, Eigen::Dynamic> points;
	vector<cv::Point2f> n_pts;
	cv::Mat mask_arc;//
	int n_max_cnt = 400;
	int MIN_DIST = 10;
	super_eventpoint->infer(image, points, n_max_cnt, n_pts, MIN_DIST, mask_arc, point_2d_uv);

	fe_points = points;
}

/**
 * @brief   extract fast feature points and calculate descriptors
 */
void KeyFrame::computeBRIEFPoint_glue(std::shared_ptr<Super_EventPoint> super_eventpoint)
{
	BriefExtractor extractor(BRIEF_PATTERN_FILE.c_str());

	Eigen::Matrix<double, 259, Eigen::Dynamic> points;
	vector<cv::Point2f> n_pts;
	cv::Mat mask_arc;
	int n_max_cnt = 400;
	int MIN_DIST = 10;

	_point_mutex.lock();
	super_eventpoint->infer(image, points, n_max_cnt, n_pts, MIN_DIST, mask_arc);
	_point_mutex.unlock();

	for(int i = 0; i < (int)n_pts.size(); i++)
	{
	    cv::KeyPoint key;
	    key.pt = n_pts[i];
	    keypoints.push_back(key); 
	}

	// // ROS_INFO("the size of keypoints in loop detection using FAST:%d",keypoints.size());
	extractor(image, keypoints, brief_descriptors);
	for (int i = 0; i < (int)keypoints.size(); i++)
	{
		Eigen::Vector3d tmp_p;
		m_camera->liftProjective(Eigen::Vector2d(keypoints[i].pt.x, keypoints[i].pt.y), tmp_p);
		cv::KeyPoint tmp_norm;
		tmp_norm.pt = cv::Point2f(tmp_p.x()/tmp_p.z(), tmp_p.y()/tmp_p.z());// normalize
		keypoints_norm.push_back(tmp_norm);
	}

	pg_points = points;
}

void BriefExtractor::operator() (const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<BRIEF::bitset> &descriptors) const
{
  m_brief.compute(im, keys, descriptors);
}


void KeyFrame::PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                         const std::vector<cv::Point3f> &matched_3d,
                         std::vector<uchar> &status,
                         Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old)
{
    cv::Mat r, rvec, t, D, tmp_r;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    Matrix3d R_inital;
    Vector3d P_inital;
    Matrix3d R_w_c = origin_vio_R * qic; 
    Vector3d T_w_c = origin_vio_T + origin_vio_R * tic;

    R_inital = R_w_c.inverse();
    P_inital = -(R_inital * T_w_c);

    cv::eigen2cv(R_inital, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_inital, t);

    cv::Mat inliers;
    TicToc t_pnp_ransac;

    if (CV_MAJOR_VERSION < 3)
        solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 100, inliers);
    else
    {
        if (CV_MINOR_VERSION < 2)
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, sqrt(10.0 / 460.0), 0.99, inliers);
        else
            solvePnPRansac(matched_3d, matched_2d_old_norm, K, D, rvec, t, true, 100, 10.0 / 460.0, 0.99, inliers);

    }

    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
        status.push_back(0);

    for( int i = 0; i < inliers.rows; i++)
    {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }

    cv::Rodrigues(rvec, r);
    Matrix3d R_pnp, R_w_c_old;
    cv::cv2eigen(r, R_pnp);
    R_w_c_old = R_pnp.transpose();
    Vector3d T_pnp, T_w_c_old;
    cv::cv2eigen(t, T_pnp);
    T_w_c_old = R_w_c_old * (-T_pnp);

    PnP_R_old = R_w_c_old * qic.transpose();
    PnP_T_old = T_w_c_old - PnP_R_old * tic;

}

bool KeyFrame::findConnection_glue(KeyFrame* old_kf, std::shared_ptr<PointMatching> super_eventmatch)
{
	TicToc tmp_t;
	//printf("find Connection\n");
	vector<cv::Point2f> matched_2d_cur;
	vector<cv::Point2f> matched_2d_cur_norm;
	vector<cv::Point3f> matched_3d;
	vector<double> matched_id;

	matched_3d = point_3d;
	matched_2d_cur = point_2d_uv;
	matched_2d_cur_norm = point_2d_norm;
	matched_id = point_id;

	std::vector<cv::Point2f> matched_2d_old(matched_2d_cur.size(), cv::Point2f(-1, -1)); 
	std::vector<cv::Point2f> matched_2d_old_norm(matched_2d_cur_norm.size(), cv::Point2f(-1, -1));
	std::vector<uchar> status(matched_2d_cur.size(), 0); 

    TicToc t_match_super;
	///////////
	std::vector<cv::DMatch> matches;
    _point_mutex.lock();
    super_eventmatch->MatchingPoints(old_kf->pg_points, fe_points, matches);//old_kf->keypoints, matched_2d_cur
    _point_mutex.unlock();

	// // std::cout << "matches.size() = " << matches.size() << std::endl;

	for (const auto &match : matches){
		int queryIdx = match.queryIdx; 
        int trainIdx = match.trainIdx;

		if (queryIdx >= 0 && queryIdx < pg_points.size() && trainIdx >= 0 && trainIdx < fe_points.size()) {
			matched_2d_old[trainIdx] = old_kf->keypoints[queryIdx].pt;
			matched_2d_old_norm[trainIdx] = old_kf->keypoints_norm[queryIdx].pt;
			status[trainIdx] = 1;
		}
	}

	TicToc t_match;

	reduceVector(matched_2d_cur, status);
	reduceVector(matched_2d_old, status);
	reduceVector(matched_2d_cur_norm, status);
	reduceVector(matched_2d_old_norm, status);
	reduceVector(matched_3d, status);
	reduceVector(matched_id, status);

	status.clear();


	Eigen::Vector3d PnP_T_old;
	Eigen::Matrix3d PnP_R_old;
	Eigen::Vector3d relative_t;
	Quaterniond relative_q;
	double relative_yaw;

	// std::cout << "matched_2d_cur = " << matched_2d_cur.size() << std::endl; //50-70

	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
		status.clear();
	    PnPRANSAC(matched_2d_old_norm, matched_3d, status, PnP_T_old, PnP_R_old);

		reduceVector(matched_2d_cur, status);
	    reduceVector(matched_2d_old, status);
	    reduceVector(matched_2d_cur_norm, status);
	    reduceVector(matched_2d_old_norm, status);
	    reduceVector(matched_3d, status);
	    reduceVector(matched_id, status);

	    #if 1
	    	if (DEBUG_IMAGE)
	        {
	        	int gap = 10;
	        	cv::Mat gap_image(ROW, gap, CV_8UC1, cv::Scalar(255, 255, 255));
	            cv::Mat gray_img, loop_match_img;
	            cv::Mat old_img = old_kf->image;
	            cv::hconcat(image, gap_image, gap_image);
	            cv::hconcat(gap_image, old_img, gray_img);
	            cvtColor(gray_img, loop_match_img, CV_GRAY2RGB);

	            for(int i = 0; i< (int)matched_2d_cur.size(); i++)
	            {
	                cv::Point2f cur_pt = matched_2d_cur[i];
	                cv::circle(loop_match_img, cur_pt, 3, cv::Scalar(0, 0, 255),-1); //3,4.5
	            }
	            for(int i = 0; i< (int)matched_2d_old.size(); i++)
	            {
	                cv::Point2f old_pt = matched_2d_old[i];
	                old_pt.x += (COL + gap);
	                cv::circle(loop_match_img, old_pt, 3, cv::Scalar(0, 0, 255),-1);
	            }
	            for (int i = 0; i< (int)matched_2d_cur.size(); i++)
	            {
	                cv::Point2f old_pt = matched_2d_old[i];
	                old_pt.x += (COL + gap) ;
	                cv::line(loop_match_img, matched_2d_cur[i], old_pt, cv::Scalar(0, 255, 255), 1, 8, 0);
	            }

	            cv::Mat notation(50, COL + gap + COL, CV_8UC3, cv::Scalar(255, 255, 255));

				putText(notation, "Current: #" + to_string(index), 
						cv::Point2f(20, 30), 
						cv::FONT_HERSHEY_DUPLEX, 
						0.9,                     
						cv::Scalar(0, 0, 0),   
						3);                    

				putText(notation, "Reference: #" + to_string(old_kf->index), 
						cv::Point2f(20 + COL + gap, 30), 
						cv::FONT_HERSHEY_DUPLEX, 
						0.9, 
						cv::Scalar(0, 0, 0),   
						3);

				cv::vconcat(notation, loop_match_img, loop_match_img);

	            if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	            {
					sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", loop_match_img).toImageMsg();
	                msg->header.stamp = ros::Time(time_stamp);
	    	    	pub_match_img.publish(msg);

					ostringstream path;
					path <<  "/home/cpy/loop_image/loop_image"
							<<"cur_frame_"<< index << "___"
							<<"pre_frame_"<< old_kf->index 
							<< "___" << "loop_image.jpg";
					// cv::imwrite( path.str().c_str(), loop_match_img);
	            }
	        }
	    #endif
	}

	if ((int)matched_2d_cur.size() > MIN_LOOP_NUM)
	{
	    relative_t = PnP_R_old.transpose() * (origin_vio_T - PnP_T_old);
	    relative_q = PnP_R_old.transpose() * origin_vio_R;
	    relative_yaw = Utility::normalizeAngle(Utility::R2ypr(origin_vio_R).x() - Utility::R2ypr(PnP_R_old).x());

	    if (abs(relative_yaw) < 30.0 && relative_t.norm() < 20.0)
	    {

	    	has_loop = true;
	    	loop_index = old_kf->index;
	    	loop_info << relative_t.x(), relative_t.y(), relative_t.z(),
	    	             relative_q.w(), relative_q.x(), relative_q.y(), relative_q.z(),
	    	             relative_yaw;
	    	if(FAST_RELOCALIZATION)
	    	{
			    sensor_msgs::PointCloud msg_match_points;
			    msg_match_points.header.stamp = ros::Time(time_stamp);
			    for (int i = 0; i < (int)matched_2d_old_norm.size(); i++)
			    {
		            geometry_msgs::Point32 p;
		            p.x = matched_2d_old_norm[i].x;
		            p.y = matched_2d_old_norm[i].y;
		            p.z = matched_id[i];
		            msg_match_points.points.push_back(p);
			    }
			    Eigen::Vector3d T = old_kf->T_w_i;
			    Eigen::Matrix3d R = old_kf->R_w_i;
			    Quaterniond Q(R);
			    sensor_msgs::ChannelFloat32 t_q_index;
			    t_q_index.values.push_back(T.x());
			    t_q_index.values.push_back(T.y());
			    t_q_index.values.push_back(T.z());
			    t_q_index.values.push_back(Q.w());
			    t_q_index.values.push_back(Q.x());
			    t_q_index.values.push_back(Q.y());
			    t_q_index.values.push_back(Q.z());
			    t_q_index.values.push_back(index);
			    msg_match_points.channels.push_back(t_q_index);
			    pub_match_points.publish(msg_match_points);
	    	}
	        return true;
	    }
	}

	return false;
}

void KeyFrame::getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = vio_T_w_i;
    _R_w_i = vio_R_w_i;
}

void KeyFrame::getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i)
{
    _T_w_i = T_w_i;
    _R_w_i = R_w_i;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
    T_w_i = _T_w_i;
    R_w_i = _R_w_i;
}

void KeyFrame::updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i)
{
	vio_T_w_i = _T_w_i;
	vio_R_w_i = _R_w_i;
	T_w_i = vio_T_w_i;
	R_w_i = vio_R_w_i;
}

Eigen::Vector3d KeyFrame::getLoopRelativeT()
{
    return Eigen::Vector3d(loop_info(0), loop_info(1), loop_info(2));
}

Eigen::Quaterniond KeyFrame::getLoopRelativeQ()
{
    return Eigen::Quaterniond(loop_info(3), loop_info(4), loop_info(5), loop_info(6));
}

double KeyFrame::getLoopRelativeYaw()
{
    return loop_info(7);
}

void KeyFrame::updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info)
{
	if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0)
	{
		//printf("update loop info\n");
		loop_info = _loop_info;
	}
}

BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;

  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;

  m_brief.importPairs(x1, y1, x2, y2);
}


