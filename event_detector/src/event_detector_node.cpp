#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>

#include "event_detector.h"

#include "dvs_msgs/Event.h"
#include "dvs_msgs/EventArray.h"
#include "utility/visualization.h"

#include <queue>
#include <thread>
#include <mutex>
#include <tuple>

#include "read_configs.h"
#include "super_eventpoint.h"
#include "global_config.h"


#define SHOW_UNDISTORTION 0

FeatureTracker trackerData[NUM_OF_CAM];
double first_image_time;
int pub_count = 1;
bool first_image_flag = true;
double last_image_time = 0;
bool init_pub = 0;

using EventQueue = std::queue<dvs_msgs::EventArray>;
EventQueue events_buf;
std::mutex m_buf;

std::string global_config_path;
std::string global_model_dir;


void pubTrackImage(const cv::Mat &imgTrack, const cv::Mat &imgTrack_two, const cv::Mat &time_surface_map, const double t)
{
    std_msgs::Header header;
    header.frame_id = "world";
    header.stamp = ros::Time(t);

    sensor_msgs::ImagePtr imgTrackMsg = cv_bridge::CvImage(header, "bgr8", imgTrack).toImageMsg();
    pub_match.publish(imgTrackMsg);

    sensor_msgs::ImagePtr imgTrackMsg_two = cv_bridge::CvImage(header, "bgr8", imgTrack_two).toImageMsg();
    pub_match_two.publish(imgTrackMsg_two);

    sensor_msgs::ImagePtr TimeSurfaceImg = cv_bridge::CvImage(header, "mono8", time_surface_map).toImageMsg();
    pub_time_surface.publish(TimeSurfaceImg);

}


void eventsCallback_buf(const dvs_msgs::EventArray &event_msg){
    m_buf.lock();
    if (!events_buf.empty())
      events_buf.pop();
    events_buf.push(event_msg);
    m_buf.unlock();
}

void handle_mono_event(const dvs_msgs::EventArray &event_msg, double msg_timestamp){

    TicToc t_whole;
    const int n_event =event_msg.events.size();
    // ROS_INFO("THE SIZE OF EVENT:%d",n_event);
    if (n_event ==0) {
        ROS_WARN("not event, please move the event camera or check whether connecting");  
        return;
    }

    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = event_msg.events[0].ts.toSec();
        last_image_time = event_msg.events[0].ts.toSec();
        return;
    }

    if (event_msg.events[0].ts.toSec() - last_image_time > 1.0 || event_msg.events[0].ts.toSec() < last_image_time)
    {
        ROS_WARN("event stream discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    last_image_time = event_msg.events[0].ts.toSec();

    if (round(1.0 * pub_count / (event_msg.events[0].ts.toSec() - first_image_time)) <= FREQ) 
    {
        PUB_THIS_FRAME = true;
        if (abs(1.0 * pub_count / (event_msg.events[0].ts.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = event_msg.events[0].ts.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    trackerData[0].readEvent(event_msg, event_msg.events[0].ts.toSec());

    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i); 
        if (!completed)
            break;
    }


   if (PUB_THIS_FRAME)
   {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = event_msg.header;
        feature_points->header.frame_id = "world";
        feature_points->header.stamp = event_msg.events[0].ts;

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts = trackerData[i].cur_un_pts;
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;
                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }

        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img.publish(feature_points);

        if (SHOW_TRACK)
        {
            cv::Mat imageTrack=trackerData[0].getTrackImage();
            cv::Mat imgTrack_two =trackerData[0].getTrackImage_two();
            cv::Mat Time_surface_map =trackerData[0].gettimesurface();
            pubTrackImage(imageTrack,imgTrack_two,Time_surface_map,last_image_time);
        }
    }
}

void sync_process();


void eventsCallback(const dvs_msgs::EventArray &event_msg)
{
    TicToc t_whole;
    const int n_event =event_msg.events.size();
    if (n_event ==0) {
        ROS_WARN("not event, please move the event camera or check whether connecting");  
        return;
    }

    if(first_image_flag)
    {
        first_image_flag = false;
        first_image_time = event_msg.events[0].ts.toSec();
        last_image_time = event_msg.events[0].ts.toSec();
        return;
    }

    if (event_msg.events[0].ts.toSec() - last_image_time > 1.0 || event_msg.events[0].ts.toSec() < last_image_time)
    {
        ROS_WARN("event stream discontinue! reset the feature tracker!");
        first_image_flag = true; 
        last_image_time = 0;
        pub_count = 1;
        std_msgs::Bool restart_flag;
        restart_flag.data = true;
        pub_restart.publish(restart_flag);
        return;
    }
    last_image_time = event_msg.events[0].ts.toSec();

    if (round(1.0 * pub_count / (event_msg.events[0].ts.toSec() - first_image_time)) <= FREQ)
    {
        PUB_THIS_FRAME = true;
        if (abs(1.0 * pub_count / (event_msg.events[0].ts.toSec() - first_image_time) - FREQ) < 0.01 * FREQ)
        {
            first_image_time = event_msg.events[0].ts.toSec();
            pub_count = 0;
        }
    }
    else
        PUB_THIS_FRAME = false;

    trackerData[0].readEvent(event_msg,event_msg.header.stamp.toSec());
    

    for (unsigned int i = 0;; i++)
    {
        bool completed = false;
        for (int j = 0; j < NUM_OF_CAM; j++)
            if (j != 1 || !STEREO_TRACK)
                completed |= trackerData[j].updateID(i);
        if (!completed)
            break;
    }

    events_buf.push(event_msg);


   if (PUB_THIS_FRAME)
   {
        pub_count++;
        sensor_msgs::PointCloudPtr feature_points(new sensor_msgs::PointCloud);
        sensor_msgs::ChannelFloat32 id_of_point;
        sensor_msgs::ChannelFloat32 u_of_point;
        sensor_msgs::ChannelFloat32 v_of_point;
        sensor_msgs::ChannelFloat32 velocity_x_of_point;
        sensor_msgs::ChannelFloat32 velocity_y_of_point;

        feature_points->header = event_msg.header;
        feature_points->header.frame_id = "world";
        feature_points->header.stamp = event_msg.events[0].ts;

        vector<set<int>> hash_ids(NUM_OF_CAM);
        for (int i = 0; i < NUM_OF_CAM; i++)
        {
            auto &un_pts = trackerData[i].cur_un_pts;
            auto &cur_pts = trackerData[i].cur_pts;
            auto &ids = trackerData[i].ids;
            auto &pts_velocity = trackerData[i].pts_velocity;
            for (unsigned int j = 0; j < ids.size(); j++)
            {
                if (trackerData[i].track_cnt[j] > 1)
                {
                    int p_id = ids[j];
                    hash_ids[i].insert(p_id);
                    geometry_msgs::Point32 p;
                    p.x = un_pts[j].x;
                    p.y = un_pts[j].y;
                    p.z = 1;
                    feature_points->points.push_back(p);
                    id_of_point.values.push_back(p_id * NUM_OF_CAM + i);
                    u_of_point.values.push_back(cur_pts[j].x);
                    v_of_point.values.push_back(cur_pts[j].y);
                    velocity_x_of_point.values.push_back(pts_velocity[j].x);
                    velocity_y_of_point.values.push_back(pts_velocity[j].y);
                }
            }
        }

        feature_points->channels.push_back(id_of_point);
        feature_points->channels.push_back(u_of_point);
        feature_points->channels.push_back(v_of_point);
        feature_points->channels.push_back(velocity_x_of_point);
        feature_points->channels.push_back(velocity_y_of_point);
        ROS_DEBUG("publish %f, at %f", feature_points->header.stamp.toSec(), ros::Time::now().toSec());
        
        if (!init_pub)
        {
            init_pub = 1;
        }
        else
            pub_img.publish(feature_points);

        if (SHOW_TRACK)
        {
            cv::Mat imageTrack=trackerData[0].getTrackImage();
            cv::Mat imgTrack_two =trackerData[0].getTrackImage_two();
            cv::Mat Time_surface_map =trackerData[0].gettimesurface();
            pubTrackImage(imageTrack,imgTrack_two,Time_surface_map,last_image_time);
        }
    }
}


int main(int argc, char **argv)
{
    ROS_WARN("into event feature detection and tracking");
    ros::init(argc, argv, "event_detector");
    ros::NodeHandle n("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
    readParameters(n);

    n.getParam("config_file", global_config_path);
    n.getParam("model_dir", global_model_dir);

    Configs configs(global_config_path, global_model_dir);

    for (int i = 0; i < NUM_OF_CAM; i++){
        trackerData[i].readIntrinsicParameter(CAM_NAMES[i]); 
        //init super_eventpoint
        trackerData[i].build_super_eventpoint(configs);
    }

    ros::Subscriber event_sub = n.subscribe(EVENT_TOPIC, 100, &eventsCallback_buf);

    registerPub(n);

    std::thread sync_thread{sync_process};

    ros::spin();
    return 0;
}

void sync_process()
{
    while(1){
        dvs_msgs::EventArray event_msg;
        double msg_timestamp_events = 0.0;

        m_buf.lock();
        if (!events_buf.empty())
        {   
            msg_timestamp_events = events_buf.front().events[0].ts.toSec();
            event_msg = events_buf.front();
            events_buf.pop();
        }
        m_buf.unlock();

        if(event_msg.events.size() != 0)
        {
            handle_mono_event(event_msg, msg_timestamp_events);
        }

        std::chrono::milliseconds dura(2);
        std::this_thread::sleep_for(dura);
    }
    
}