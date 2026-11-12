#
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#

# Picks the default macOS deployment target for the mlx presets. Runs before the
# first project() via CMAKE_PROJECT_INCLUDE_BEFORE, since the deployment target
# must be set before languages are enabled.
#
# MLX compiles its NAX kernels only for a deployment target of 26.2 or newer,
# built with a 26.2+ SDK. A 26.2 build also requires a 26.2+ host at runtime:
# MLX picks the Metal language version with __builtin_available, which a 26.2
# deployment target resolves at compile time. So use 26.2 only when both the SDK
# and this host support it, and otherwise fall back to MLX's 14.0 minimum. Pass
# -DCMAKE_OSX_DEPLOYMENT_TARGET to override.
if(DEFINED CMAKE_OSX_DEPLOYMENT_TARGET)
  return()
endif()

execute_process(
  COMMAND xcrun --sdk macosx --show-sdk-version
  OUTPUT_VARIABLE _mlx_sdk_version
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
)
execute_process(
  COMMAND sw_vers -productVersion
  OUTPUT_VARIABLE _mlx_host_version
  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
)

if(_mlx_sdk_version VERSION_GREATER_EQUAL 26.2 AND _mlx_host_version
                                                   VERSION_GREATER_EQUAL 26.2
)
  set(_mlx_default_deployment_target 26.2)
else()
  set(_mlx_default_deployment_target 14.0)
endif()

message(
  STATUS
    "MLX: defaulting CMAKE_OSX_DEPLOYMENT_TARGET to ${_mlx_default_deployment_target} "
    "(macOS ${_mlx_host_version}, SDK ${_mlx_sdk_version})"
)
set(CMAKE_OSX_DEPLOYMENT_TARGET
    ${_mlx_default_deployment_target}
    CACHE STRING "Minimum macOS version to target"
)
