# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Runs the add/multiply model in main/ on its bundled test case through the
# et_model C API and checks the outputs. main/smoke_kernels is the model's
# kernel bundle, made with this component's tools/gen_kernels.py.

cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(hello_executorch)
