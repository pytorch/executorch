/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/Logging.h>
#include <executorch/backends/qualcomm/runtime/Utils.h>
#include <sys/stat.h>
namespace torch {
namespace executor {
namespace qnn {

void CreateDirectory(const std::string& path) {
  // Create any recursive directory
  if (path.empty()) {
    QNN_EXECUTORCH_LOG_ERROR("Create folder shouldn't be empty");
    return;
  }
  std::size_t pos = path.find_last_of('/');
  std::string subdir = (std::string::npos == pos) ? "" : path.substr(0, pos);
  if (subdir.empty() || subdir == "." || subdir == "..") {
    return;
  }
  CreateDirectory(subdir);
  int mkdir_err = mkdir(subdir.c_str(), S_IRWXU | S_IRWXG | S_IRWXO);
  if (mkdir_err != 0 && errno != EEXIST) {
    std::string err_msg = "Failed to create " + subdir + " folder\n";
    QNN_EXECUTORCH_LOG_ERROR(err_msg.c_str());
  }
}

} // namespace qnn
} // namespace executor
} // namespace torch
