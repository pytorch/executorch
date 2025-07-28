/*
 * Copyright (c) Qualcomm Innovation Center, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once
#include <mutex>
#include <string>
#include <unordered_set>

namespace executorch {
namespace backends {
namespace qnn {
class QnnOpPackageManager {
 public:
  QnnOpPackageManager() = default;
  ~QnnOpPackageManager() = default;

  QnnOpPackageManager(const QnnOpPackageManager& rhs) = delete;
  QnnOpPackageManager(QnnOpPackageManager&& rhs) = delete;
  QnnOpPackageManager& operator=(const QnnOpPackageManager& rhs) = delete;
  QnnOpPackageManager& operator=(QnnOpPackageManager&& rhs) = delete;

  bool Add(std::string qnn_op_name);

  bool Has(std::string qnn_op_name);

  bool Erase(std::string qnn_op_name);

  void Clear();

 private:
  std::unordered_set<std::string> qnn_op_package_path_set_;
  std::mutex table_mutex_;
};

} // namespace qnn
} // namespace backends
} // namespace executorch
