# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Main component CMakeLists.txt for the ESP-IDF project. This is a minimal main
# component that depends on the executor_runner.

idf_component_register(
  SRCS "main.cpp" INCLUDE_DIRS "." REQUIRES executor_runner
)
