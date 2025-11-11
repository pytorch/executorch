/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <executorch/backends/vulkan/runtime/api/api.h>
#include <folly/json.h>
#include <fstream>
#include <iostream>

#include "utils.h"

namespace gpuinfo {

class App {
 private:
  folly::dynamic config_;

 public:
  size_t buf_cache_size;
  uint32_t max_shared_mem_size;
  uint32_t sm_count;
  uint32_t nthread_logic;
  uint32_t subgroup_size;
  uint32_t max_tex_width;
  uint32_t max_tex_height;
  uint32_t max_tex_depth;

  App() {
    context()->initialize_querypool();

    std::cout << context()->adapter_ptr()->stringize() << std::endl
              << std::endl;

    auto cl_device = get_cl_device();

    sm_count = cl_device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>();
    nthread_logic = cl_device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>();
    buf_cache_size = cl_device.getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>();
    max_shared_mem_size = cl_device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>();
    max_tex_width = cl_device.getInfo<CL_DEVICE_IMAGE3D_MAX_WIDTH>();
    max_tex_height = cl_device.getInfo<CL_DEVICE_IMAGE3D_MAX_HEIGHT>();
    max_tex_depth = cl_device.getInfo<CL_DEVICE_IMAGE3D_MAX_DEPTH>();

    VkPhysicalDeviceSubgroupProperties subgroup_props{};
    VkPhysicalDeviceProperties2 props2{};

    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroup_props;
    subgroup_props.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    vkGetPhysicalDeviceProperties2(
        context()->adapter_ptr()->physical_handle(), &props2);
    subgroup_size = subgroup_props.subgroupSize;

    std::cout << std::endl;
    std::cout << "SM count," << sm_count << std::endl;
    std::cout << "Logic Thread Count," << nthread_logic << std::endl;
    std::cout << "Cache Size," << buf_cache_size << std::endl;
    std::cout << "Shared Memory Size," << max_shared_mem_size << std::endl;
    std::cout << "SubGroup Size," << subgroup_size << std::endl;
    std::cout << "MaxTexWidth," << max_tex_width << std::endl;
    std::cout << "MaxTexHeight," << max_tex_height << std::endl;
    std::cout << "MaxTexDepth," << max_tex_depth << std::endl;
  }

  float get_config(const std::string& test, const std::string& key) const {
    if (config_[test].empty()) {
      throw std::runtime_error("Missing config for " + test);
    }

    if (!config_[test][key].isNumber()) {
      throw std::runtime_error(
          "Config for " + test + "." + key + " is not a number");
    }

    float value;
    if (config_[test][key].isDouble()) {
      value = config_[test][key].getDouble();
    } else {
      value = config_[test][key].getInt();
    }

    std::cout << "Read value for " << test << "." << key << " = " << value
              << std::endl;
    return value;
  }

  bool enabled(const std::string& test) const {
    if (config_.empty() || config_[test].empty() ||
        !config_[test]["enabled"].isBool()) {
      return true;
    }
    return config_[test]["enabled"].getBool();
  }

  void load_config(std::string file_path) {
    std::ifstream file(file_path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string json_str = buffer.str();
    if (json_str.empty()) {
      throw std::runtime_error(
          "Failed to read config file from " + file_path + ".");
    }
    config_ = folly::parseJson(json_str);
  }
};
} // namespace gpuinfo
