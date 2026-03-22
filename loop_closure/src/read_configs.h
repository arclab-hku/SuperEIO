#ifndef READ_CONFIGS_H_
#define READ_CONFIGS_H_

#include <iostream>
#include <yaml-cpp/yaml.h>

#include "utils.h"

struct Super_EventPointConfig {
  int max_keypoints;
  double keypoint_threshold;
  int remove_borders;
  int dla_core;
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};

struct Super_EventMatchConfig {
  int image_width;
  int image_height;
  int dla_core;
  std::vector<std::string> input_tensor_names;
  std::vector<std::string> output_tensor_names;
  std::string onnx_file;
  std::string engine_file;
};


struct Configs{
  std::string dataroot;
  std::string model_dir;

  Super_EventPointConfig super_eventpoint_config;
  Super_EventMatchConfig super_eventmatch_config;

  Configs(const std::string& config_file, const std::string& model_dir){
    // std::cout << "config_file = " << config_file << std::endl;
    if(!FileExists(config_file)){
      std::cout << "config file: " << config_file << " doesn't exist" << std::endl;
      return;
    }
    YAML::Node file_node = YAML::LoadFile(config_file);

    YAML::Node super_eventpoint_node = file_node["super_eventpoint"];
    super_eventpoint_config.max_keypoints = super_eventpoint_node["max_keypoints"].as<int>();
    super_eventpoint_config.keypoint_threshold = super_eventpoint_node["keypoint_threshold"].as<double>();
    super_eventpoint_config.remove_borders = super_eventpoint_node["remove_borders"].as<int>();
    super_eventpoint_config.dla_core = super_eventpoint_node["dla_core"].as<int>();
    YAML::Node super_eventpoint_input_tensor_names_node = super_eventpoint_node["input_tensor_names"];
    size_t super_eventpoint_num_input_tensor_names = super_eventpoint_input_tensor_names_node.size();
    for(size_t i = 0; i < super_eventpoint_num_input_tensor_names; i++){
      super_eventpoint_config.input_tensor_names.push_back(super_eventpoint_input_tensor_names_node[i].as<std::string>());
    }
    YAML::Node super_eventpoint_output_tensor_names_node = super_eventpoint_node["output_tensor_names"];
    size_t super_eventpoint_num_output_tensor_names = super_eventpoint_output_tensor_names_node.size();
    for(size_t i = 0; i < super_eventpoint_num_output_tensor_names; i++){
      super_eventpoint_config.output_tensor_names.push_back(super_eventpoint_output_tensor_names_node[i].as<std::string>());
    }
    std::string super_eventpoint_onnx_file = super_eventpoint_node["onnx_file"].as<std::string>();
    std::string super_eventpoint_engine_file= super_eventpoint_node["engine_file"].as<std::string>();
    super_eventpoint_config.onnx_file = ConcatenateFolderAndFileName(model_dir, super_eventpoint_onnx_file);
    super_eventpoint_config.engine_file = ConcatenateFolderAndFileName(model_dir, super_eventpoint_engine_file);
    
    YAML::Node super_eventmatch_node = file_node["super_eventmatch"];
    super_eventmatch_config.image_width = super_eventmatch_node["image_width"].as<int>();
    super_eventmatch_config.image_height = super_eventmatch_node["image_height"].as<int>();
    super_eventmatch_config.dla_core = super_eventmatch_node["dla_core"].as<int>();
    YAML::Node super_eventmatch_input_tensor_names_node = super_eventmatch_node["input_tensor_names"];
    size_t super_eventmatch_num_input_tensor_names = super_eventmatch_input_tensor_names_node.size();
    for(size_t i = 0; i < super_eventmatch_num_input_tensor_names; i++){
      super_eventmatch_config.input_tensor_names.push_back(super_eventmatch_input_tensor_names_node[i].as<std::string>());
    }
    YAML::Node super_eventmatch_output_tensor_names_node = super_eventmatch_node["output_tensor_names"];
    size_t super_eventmatch_num_output_tensor_names = super_eventmatch_output_tensor_names_node.size();
    for(size_t i = 0; i < super_eventmatch_num_output_tensor_names; i++){
      super_eventmatch_config.output_tensor_names.push_back(super_eventmatch_output_tensor_names_node[i].as<std::string>());
    }
    std::string super_eventmatch_onnx_file = super_eventmatch_node["onnx_file"].as<std::string>();
    std::string super_eventmatch_engine_file= super_eventmatch_node["engine_file"].as<std::string>();
    super_eventmatch_config.onnx_file = ConcatenateFolderAndFileName(model_dir, super_eventmatch_onnx_file);
    super_eventmatch_config.engine_file = ConcatenateFolderAndFileName(model_dir, super_eventmatch_engine_file); 
  }
};

#endif  // READ_CONFIGS_H_
