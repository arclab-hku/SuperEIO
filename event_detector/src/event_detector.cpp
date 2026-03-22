#include "event_detector.h"
#include "TS_generator/TS_generator.h"
#include <thread>

TS::TSGenerator generator = TS::TSGenerator();

int FeatureTracker::n_id = 0;
int frame_index = 0;

bool inBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

void reduceVector(vector<int> &v, vector<uchar> status)
{
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (status[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker()
{
}

void FeatureTracker::build_super_eventpoint(Configs& configs) {
    super_eventpoint = std::shared_ptr<Super_EventPoint>(new Super_EventPoint(configs.super_eventpoint_config));
    if (!super_eventpoint->build()) {
        ROS_ERROR("Error in super_eventpoint building");
        exit(0); 
    }
    std::cout << "build super_eventpoint" << std::endl;
}


void FeatureTracker::setevent_Mask()
{
    mask_arc = cv::Mat(ROW, COL, CV_64FC1, cv::Scalar(0.0));

    vector<pair<int, pair<cv::Point2f, int>>> cnt_pts_id;

    for (unsigned int i = 0; i < cur_pts.size(); i++)
        cnt_pts_id.push_back(make_pair(track_cnt[i], make_pair(cur_pts[i], ids[i])));

    sort(cnt_pts_id.begin(), cnt_pts_id.end(), [](const pair<int, pair<cv::Point2f, int>> &a, const pair<int, pair<cv::Point2f, int>> &b)
         {
            return a.first > b.first;
         });

    cur_pts.clear();
    ids.clear();
    track_cnt.clear();

    for (auto &it : cnt_pts_id)
    {
        if (mask_arc.at<double>(it.second.first) == 0.0)
        {
            cur_pts.push_back(it.second.first);
            ids.push_back(it.second.second);
            track_cnt.push_back(it.first);
            cv::circle(mask_arc, it.second.first, MIN_DIST, 255.0, -1);
        }
    }
}


void FeatureTracker::addPoints()
{
    for (auto &p : n_pts)
    {
        forw_pts.push_back(p);
        ids.push_back(-1);
        track_cnt.push_back(1);
    }
}


cv::Mat FeatureTracker::getTrackImage()
{
    return imTrack;
}

cv::Mat FeatureTracker::getTrackImage_two()
{
    return imTrack_two;
}

cv::Mat FeatureTracker::gettimesurface()
{
    return time_surface_visualization;
}


void FeatureTracker::readEvent(const dvs_msgs::EventArray &last_event, double _cur_time)
{
    cv::Mat img;

    cur_time = _cur_time;

    if(FLAG_DETECTOR_NOSTART){
        FLAG_DETECTOR_NOSTART=false;
        generator.init(COL,ROW);
    }

    generator.cur_event_mat=cv::Mat::zeros(cv::Size(COL, ROW), CV_8UC3);

    for (const dvs_msgs::Event& e:last_event.events){
        generator.createSAE(e.ts.toSec(), e.x, e.y, e.polarity);
    }

    cv::Mat event_mat=generator.cur_event_mat;

    cv::Mat raw_event=generator.cur_event_mat;
    cv::cvtColor(raw_event, raw_event, cv::COLOR_BGR2GRAY);

    TicToc t_ts;
    const cv::Mat time_surface_map=generator.SAEtoTimeSurface(cur_time);

    time_surface_visualization=time_surface_map.clone();
    cv::Mat time_surface = time_surface_map.clone();

    if (EQUALIZE)
    {  
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
        clahe->apply(time_surface, img);
        cv::normalize(img, img, 0, 255, CV_MINMAX);
    }
    else
        img = time_surface;


    if (cur_img.empty()) 
    {
        prev_img = cur_img= img;
    }
    else
    {
        cur_img=img;
    }

    cv::Mat rightImg;
    cur_pts.clear();

    TicToc t_tracking;
    if (prev_pts.size() > 0)
    {
        TicToc t_o;
        vector<uchar> status;
        vector<float> err;

        cv::calcOpticalFlowPyrLK(prev_img, cur_img, prev_pts, cur_pts, status, err, cv::Size(21, 21), 2);

        if(FLOW_BACK)
        {
            vector<uchar> reverse_status;
            vector<cv::Point2f> reverse_pts = prev_pts;

            cv::calcOpticalFlowPyrLK(cur_img, prev_img, cur_pts, reverse_pts, reverse_status, err, cv::Size(21, 21), 2);
            for(size_t i = 0; i < status.size(); i++)
            {
                if(status[i] && reverse_status[i] && distance(prev_pts[i], reverse_pts[i]) <= 1)
                {
                    status[i] = 1;
                }
                else
                    status[i] = 0;
            }
        }

        for (int i = 0; i < int(cur_pts.size()); i++)
            if (status[i] && !inBorder(cur_pts[i]))
                status[i] = 0;
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
    }
   
    for (auto &n : track_cnt)
        n++;

    if (PUB_THIS_FRAME)
    {
        rejectWithF_event();
        ROS_DEBUG("set mask begins");
        TicToc t_m;
        setevent_Mask();
        ROS_DEBUG("set mask costs %fms", t_m.toc());

        ROS_DEBUG("detect feature begins");

        int n_max_cnt = MAX_CNT - static_cast<int>(cur_pts.size());
        if (n_max_cnt > 0)
        {
            if(mask_arc.empty())
                cout << "mask is empty!" << endl;
            if (mask_arc.type() != CV_64FC1)
                cout << "mask type wrong!" << endl;
            if (mask_arc.size() != time_surface.size())
                cout << "wrong size!" << endl;

            Eigen::Matrix<double, 259, Eigen::Dynamic> points;
            super_eventpoint->infer(time_surface, points, n_max_cnt, n_pts, MIN_DIST, mask_arc);
        }
        else
            n_pts.clear();

         for (auto &p : n_pts)
        {
            cur_pts.push_back(p);
            ids.push_back(-1);
            track_cnt.push_back(1);
        }

    }

    cur_un_pts = undistortedPts(cur_pts, m_camera);
    pts_velocity = ptsVelocity(ids, cur_un_pts, cur_un_pts_map, prev_un_pts_map);

    if(SHOW_TRACK)
    {
        event_drawTrack(cur_img, rightImg, ids, cur_pts, cur_right_pts, prevLeftPtsMap);
        event_drawTrack_two(cur_img, prev_img, ids, cur_pts, prev_pts, prevLeftPtsMap);
    }

    prev_img = cur_img;
    prev_pts = cur_pts;
    prev_un_pts = cur_un_pts;
    prev_un_pts_map = cur_un_pts_map;
    prev_time = cur_time;

    prevLeftPtsMap.clear();
    for(size_t i = 0; i < cur_pts.size(); i++)
        prevLeftPtsMap[ids[i]] = cur_pts[i];

}


void FeatureTracker::rejectWithF_event()
{
    if (cur_pts.size() >= 8)
    {
        ROS_DEBUG("FM ransac begins");
        TicToc t_f;
        vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_prev_pts(prev_pts.size());
        for (unsigned int i = 0; i < prev_pts.size(); i++)
        {
            Eigen::Vector3d tmp_p;

            m_camera->liftProjective(Eigen::Vector2d(prev_pts[i].x, prev_pts[i].y), tmp_p);

            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_prev_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);

            tmp_p.x() = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
            tmp_p.y() = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        vector<uchar> status;

        cv::findFundamentalMat(un_prev_pts, un_cur_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        int size_a = prev_pts.size();
        reduceVector(prev_pts, status);
        reduceVector(cur_pts, status);
        reduceVector(cur_un_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
        ROS_DEBUG("FM ransac costs: %fms", t_f.toc());
    }
}



/**
 * @brief 
 * 
 * @param[in] i 
 * @return true 
 * @return false
 */
bool FeatureTracker::updateID(unsigned int i)
{
    if (i < ids.size())
    {
        if (ids[i] == -1)
            ids[i] = n_id++;
        return true;
    }
    else
        return false;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file)
{
    ROS_INFO("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}


vector<cv::Point2f> FeatureTracker::undistortedPts(vector<cv::Point2f> &pts, camodocal::CameraPtr cam)
{
    vector<cv::Point2f> un_pts;
    for (unsigned int i = 0; i < pts.size(); i++)
    {
        Eigen::Vector2d a(pts[i].x, pts[i].y);
        Eigen::Vector3d b;
        cam->liftProjective(a, b);
        un_pts.push_back(cv::Point2f(b.x() / b.z(), b.y() / b.z()));
    }
    return un_pts;
}

vector<cv::Point2f> FeatureTracker::ptsVelocity(vector<int> &ids, vector<cv::Point2f> &pts, 
                                            map<int, cv::Point2f> &cur_id_pts, map<int, cv::Point2f> &prev_id_pts)
{
    vector<cv::Point2f> pts_velocity;
    cur_id_pts.clear();
    for (unsigned int i = 0; i < ids.size(); i++)
    {
        cur_id_pts.insert(make_pair(ids[i], pts[i]));
    }

    // caculate points velocity
    if (!prev_id_pts.empty())
    {
        double dt = cur_time - prev_time;
        
        for (unsigned int i = 0; i < pts.size(); i++)
        {
            std::map<int, cv::Point2f>::iterator it;
            it = prev_id_pts.find(ids[i]);
            if (it != prev_id_pts.end())
            {
                double v_x = (pts[i].x - it->second.x) / dt;
                double v_y = (pts[i].y - it->second.y) / dt;
                pts_velocity.push_back(cv::Point2f(v_x, v_y));
            }
            else
                pts_velocity.push_back(cv::Point2f(0, 0));

        }
    }
    else
    {
        for (unsigned int i = 0; i < cur_pts.size(); i++)
        {
            pts_velocity.push_back(cv::Point2f(0, 0));
        }
    }
    return pts_velocity;
}


void  FeatureTracker::event_drawTrack(const cv::Mat &imLeft, const cv::Mat &imRight, 
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts, 
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap)
{
    int cols = imLeft.cols;
    if (!imRight.empty())
        cv::hconcat(imLeft, imRight, imTrack);
    else
        imTrack = imLeft.clone();
    cv::cvtColor(imTrack, imTrack, CV_GRAY2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        if(track_cnt[j]>=2)
            cv::circle(imTrack, curLeftPts[j], 3, cv::Scalar(0, 0, 255),  -1);
        else
            cv::circle(imTrack, curLeftPts[j], 3, cv::Scalar(0, 255, 0),  1);
    }

    map<int, cv::Point2f>::iterator mapIt;
    for (size_t i = 0; i < curLeftIds.size(); i++)
    {
        int id = curLeftIds[i];
        mapIt = prevLeftPtsMap.find(id);
        if(mapIt != prevLeftPtsMap.end())
        {
            if(track_cnt[i]>=2){
                Vector2d tmp_cur_un_pts (cur_un_pts[i].x, cur_un_pts[i].y);
                Vector2d tmp_pts_velocity (pts_velocity[i].x, pts_velocity[i].y);
                Vector3d tmp_prev_un_pts;
                tmp_prev_un_pts.head(2) = tmp_cur_un_pts - 0.10 * tmp_pts_velocity;
                tmp_prev_un_pts.z() = 1;
                Vector2d tmp_prev_uv;
                m_camera->spaceToPlane(tmp_prev_un_pts, tmp_prev_uv);
            }
        }
    }
}


void  FeatureTracker::event_drawTrack_two(const cv::Mat &imLeft, const cv::Mat &imRight, 
                               vector<int> &curLeftIds,
                               vector<cv::Point2f> &curLeftPts, 
                               vector<cv::Point2f> &curRightPts,
                               map<int, cv::Point2f> &prevLeftPtsMap)
{
    int cols = imLeft.cols;
    if (!imRight.empty())
        cv::hconcat(imLeft, imRight, imTrack_two);
    else
        imTrack_two = imLeft.clone();
    cv::cvtColor(imTrack_two, imTrack_two, CV_GRAY2RGB);

    for (size_t j = 0; j < curLeftPts.size(); j++)
    {
        if(track_cnt[j]>=2)
            cv::circle(imTrack_two, curLeftPts[j], MIN_DIST/2, cv::Scalar(0, 0, 255), -1);
    }
    
    if (!imRight.empty())
    {
        map<int, cv::Point2f>::iterator mapIt;
        for (size_t i = 0; i < curLeftIds.size(); i++)
        {
            int id = curLeftIds[i];
            mapIt = prevLeftPtsMap.find(id);
            cv::Point2f rightPt = mapIt->second;
            rightPt.x += cols;
            if(mapIt != prevLeftPtsMap.end()){
                cv::circle(imTrack_two, rightPt, MIN_DIST/2, cv::Scalar(0, 255,255), -1);
                cv::Point2f leftPt = curLeftPts[i];
                if(track_cnt[i]>=2){
                    cv::arrowedLine(imTrack_two, leftPt, rightPt, cv::Scalar(0, 255, 0), 1, 8, 0, 0.02);
                }
            }
        }
    }
}

double FeatureTracker::distance(cv::Point2f &pt1, cv::Point2f &pt2)
{
    //printf("pt1: %f %f pt2: %f %f\n", pt1.x, pt1.y, pt2.x, pt2.y);
    double dx = pt1.x - pt2.x;
    double dy = pt1.y - pt2.y;
    return sqrt(dx * dx + dy * dy);
}
