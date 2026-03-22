#! /bin/bash

gnome-terminal --tab -e 'bash -c "roscore;exec bash"'
sleep 1s

#################*********************hku mono ********************************##############
# gnome-terminal --tab -e 'bash -c "roslaunch supereio_ba hku_mono.launch;exec bash"'

# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/vicon_hdr1.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/vicon_hdr2.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/vicon_darktolight1.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/vicon_lighttodark1.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/vicon_dark1.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/vicon_dark2.bag;exec bash"'


#################********************* vector ********************************##############

# gnome-terminal --tab -e 'bash -c "roslaunch supereio_ba vector.launch;exec bash"'
# gnome-terminal --tab -e 'bash -c "roslaunch supereio_ba vector_large.launch;exec bash"'

# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/desk_normal1.synced.merged.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/sofa_normal1.synced.merged.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/corner_slow1.synced.merged.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.5 -r 0.5 /home/cpy/Datasets/robot_fast1.synced.merged.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/desk_fast1.synced.merged.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 -r 0.5 /home/cpy/Datasets/mountain_fast1.synced.merged.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 -r 0.7 /home/cpy/Datasets/hdr_fast1.synced.merged.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/corridors_dolly1.synced.merged.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/corridors_walk1.synced.merged.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/school_scooter1.synced.merged.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.5 /home/cpy/Datasets/units_scooter1.synced.merged.bag;exec bash"'


#################*********************240c********************************##############
# gnome-terminal --tab -e 'bash -c "roslaunch supereio_ba 240c.launch;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/boxes_6dof.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/boxes_rotation.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/boxes_translation.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/dynamic_rotation.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.5 /home/cpy/Datasets/dynamic_translation.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/dynamic_6dof.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/hdr_boxes.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 1.0 /home/cpy/Datasets/hdr_poster.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.5 /home/cpy/Datasets/poster_6dof.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/poster_rotation.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/poster_translation.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/shapes_6dof.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/shapes_rotation.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 -r 0.7 /home/cpy/Datasets/shapes_translation.bag;exec bash"'



#################*********************hku stereo********************************##############
# gnome-terminal --tab -e 'bash -c "roslaunch evio davis_open.launch;exec bash"'
# sleep 3s
gnome-terminal --tab -e 'bash -c "roslaunch supereio_ba hku_stereo.launch;exec bash"'
# gnome-terminal --tab -e 'bash -c "roslaunch supereio_ba hku_outdoor.launch;exec bash"'
# # # ######################data set for hku davis346
gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 2.0 /home/cpy/Datasets/ESVIO/HKU_aggressive_translation.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.5 /home/cpy/Datasets/ESVIO/HKU_aggressive_rotation.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 5.5 /home/cpy/Datasets/ESVIO/HKU_aggressive_small_flip.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 1.0 /home/cpy/Datasets/ESVIO/hku_aggressive_walk.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/ESVIO/HKU_HDR_circle.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 4.0 /home/cpy/Datasets/ESVIO/HKU_HDR_slow.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 2.0 /home/cpy/Datasets/ESVIO/hku_hdr_tran_rota.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 0.0 /home/cpy/Datasets/ESVIO/hku_hdr_agg.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 2.0 /home/cpy/Datasets/ESVIO/hku_dark_normal.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 45.0 /home/cpy/Datasets/2022-12-02-18-18-41_hdr_flight_little_light.bag;exec bash"'
# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 50.0  /home/cpy/Datasets/2022-12-02-20-58-30_change_yaw_1.5.bag;exec bash"'

# gnome-terminal --window -e 'bash -c "rosbag play --pause --clock -s 5.0 /home/cpy/Datasets/mainbuilding2025-06-06-14-01-41.bag;exec bash"'
