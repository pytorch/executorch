# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Backends
define_overridable_config(EXECUTORCH_BUILD_XNNPACK "Build the XNNPACK backend" OFF)
define_overridable_config(EXECUTORCH_BUILD_COREML "Enable CoreML backend" OFF)
define_overridable_config(EXECUTORCH_BUILD_MPS "Build the MPS backend" OFF)

# Targets
define_overridable_config(EXECUTORCH_BUILD_PYBIND "Build the Python binding" OFF)
define_overridable_config(EXECUTORCH_BUILD_EXECUTOR_RUNNER "Build the executor_runner executable" OFF)
define_overridable_config(EXECUTORCH_BUILD_EXTENSION_TENSOR "Build the Tensor extension" OFF)

# Validations

if(EXECUTORCH_BUILD_PYBIND)
  if(NOT EXECUTORCH_BUILD_EXTENSION_TENSOR)
    message(FATAL_ERROR "EXECUTORCH_BUILD_EXTENSION_TENSOR must be on if EXECUTORCH_BUILD_PYBIND is on")
  endif()
endif()
