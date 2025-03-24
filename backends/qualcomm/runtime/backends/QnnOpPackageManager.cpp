/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#include <executorch/backends/qualcomm/runtime/backends/QnnOpPackageManager.h>
namespace executorch {
namespace backends {
namespace qnn {

bool QnnOpPackageManager::Add(std::string qnn_op_name) {
  const std::lock_guard<std::mutex> lock(table_mutex_);
  std::pair<decltype(qnn_op_package_path_set_)::iterator, bool> ret =
      qnn_op_package_path_set_.emplace(qnn_op_name);
  return ret.second;
}

bool QnnOpPackageManager::Has(std::string qnn_op_name) {
  const std::lock_guard<std::mutex> lock(table_mutex_);
  return qnn_op_package_path_set_.count(qnn_op_name) > 0;
}

bool QnnOpPackageManager::Erase(std::string qnn_op_name) {
  const std::lock_guard<std::mutex> lock(table_mutex_);
  return qnn_op_package_path_set_.erase(qnn_op_name) > 0;
}

void QnnOpPackageManager::Clear() {
  const std::lock_guard<std::mutex> lock(table_mutex_);
  qnn_op_package_path_set_.clear();
};

} // namespace qnn
} // namespace backends
} // namespace executorch
