#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

install_swiftshader() {
  _https_amazon_aws=https://ossci-android.s3.amazonaws.com
  _swiftshader_archive=swiftshader-abe07b943-prebuilt.tar.gz
  _swiftshader_dir=/tmp/swiftshader
  mkdir -p $_swiftshader_dir

  _tmp_archive="/tmp/${_swiftshader_archive}"

  curl --silent --show-error --location --fail --retry 3 --retry-all-errors \
    --output "${_tmp_archive}" "$_https_amazon_aws/${_swiftshader_archive}"

  tar -C "${_swiftshader_dir}" -xzf "${_tmp_archive}"

  export VK_ICD_FILENAMES="${_swiftshader_dir}/swiftshader/build/Linux/vk_swiftshader_icd.json"
  export LD_LIBRARY_PATH="${_swiftshader_dir}/swiftshader/build/Linux/:${LD_LIBRARY_PATH:-}"
  export ETVK_USING_SWIFTSHADER=1
}

install_vulkan_sdk() {
  VULKAN_SDK_VERSION=$1
  _vulkan_sdk_url="https://sdk.lunarg.com/sdk/download/${VULKAN_SDK_VERSION}/linux/vulkansdk-linux-x86_64-${VULKAN_SDK_VERSION}.tar.xz"

  _vulkan_sdk_dir=/tmp/vulkansdk
  mkdir -p $_vulkan_sdk_dir

  _tmp_archive="/tmp/vulkansdk.tar.gz"

  curl --silent --show-error --location --fail --retry 3 --retry-all-errors \
    --output "${_tmp_archive}" "${_vulkan_sdk_url}"

  tar -C "${_vulkan_sdk_dir}" -xJf "${_tmp_archive}"

  export PATH="${PATH}:${_vulkan_sdk_dir}/${VULKAN_SDK_VERSION}/x86_64/bin/"
}

_maybe_sudo() {
  if [ "$(id -u)" -eq 0 ]; then
    "$@"
  else
    sudo "$@"
  fi
}

install_glslc() {
  # The glslc shipped in the LunarG SDK is dynamically linked against a newer
  # glibc/libstdc++ than the manylinux_2_28 / AlmaLinux 8 CUDA runner image
  # provides (glibc 2.28), where it fails to load with "GLIBC_2.29 not found".
  # conda-forge's shaderc is built against an old sysroot, runs there, and is
  # recent enough for the GL_EXT_integer_dot_product / GL_KHR_cooperative_matrix
  # extensions the Vulkan shaders use. Install it into an isolated prefix so the
  # base conda env that builds ExecuTorch is left untouched, then put it on PATH.
  _glslc_prefix=/tmp/shaderc
  conda create -y -p "${_glslc_prefix}" -c conda-forge shaderc
  export PATH="${_glslc_prefix}/bin:${PATH}"
}

install_vulkan_loader() {
  # libvulkan.so.1 (the Khronos loader that volk dlopen()s at runtime) is not part
  # of the NVIDIA driver and is absent from the CUDA builder image; vulkan-tools
  # provides vulkaninfo for the device sanity check. Both ship as native el8 RPMs.
  if command -v dnf >/dev/null 2>&1; then
    _maybe_sudo dnf install -y vulkan-loader vulkan-tools
  fi
}

_find_nvidia_vulkan_library() {
  # NVIDIA implements its Vulkan ICD inside libGLX_nvidia.so.0. The NVIDIA
  # container runtime mounts this library into the container (it is pulled from
  # the driver's ldcache when NVIDIA_DRIVER_CAPABILITIES includes graphics/all),
  # so prefer ldconfig and fall back to the usual mount locations.
  local lib cand
  lib="$(ldconfig -p 2>/dev/null | awk '/libGLX_nvidia\.so\.0/ {print $NF; exit}')"
  if [ -z "${lib}" ]; then
    for cand in /usr/lib64/libGLX_nvidia.so.0 \
        /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0 \
        /usr/lib/libGLX_nvidia.so.0; do
      if [ -e "${cand}" ]; then
        lib="${cand}"
        break
      fi
    done
  fi
  printf '%s' "${lib}"
}

_vulkan_has_real_device() {
  # True if the loader enumerates a hardware GPU. vulkaninfo can exit non-zero
  # for unrelated reasons (no display/WSI), so key off the reported deviceType.
  command -v vulkaninfo >/dev/null 2>&1 || return 0
  vulkaninfo --summary 2>/dev/null |
    grep -qE 'PHYSICAL_DEVICE_TYPE_(DISCRETE|INTEGRATED|VIRTUAL)_GPU'
}

setup_real_gpu_icd() {
  # Select a Vulkan ICD so the runtime exercises the real GPU when one is usable.
  # Two quirks of the CUDA CI image make this non-trivial:
  #   1. The NVIDIA container runtime mounts the driver's Vulkan library but does
  #      not register its ICD manifest, so the loader never discovers the GPU on
  #      its own. We synthesize the manifest and pin the loader to it.
  #   2. Installing vulkan-loader/vulkan-tools pulls in mesa-vulkan-drivers,
  #      which drop Intel/AMD/lavapipe manifests for absent hardware. lavapipe
  #      fails vkCreateInstance on this image and, because the loader walks every
  #      manifest in icd.d, that poisons device enumeration for the whole
  #      process. Pinning VK_ICD_FILENAMES makes the loader ignore icd.d, so the
  #      broken stubs cannot interfere.
  local nvidia_lib
  nvidia_lib="$(_find_nvidia_vulkan_library)"
  if [ -n "${nvidia_lib}" ]; then
    local icd=/tmp/nvidia_icd.json
    cat >"${icd}" <<JSON
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "${nvidia_lib}",
        "api_version": "1.3.0"
    }
}
JSON
    export VK_ICD_FILENAMES="${icd}"
    unset ETVK_USING_SWIFTSHADER || true
    if _vulkan_has_real_device; then
      echo "Real NVIDIA GPU selected; pinned Vulkan ICD to ${nvidia_lib}"
      return
    fi
    echo "WARNING: ${nvidia_lib} present but no GPU enumerated; using SwiftShader."
    # Surface why the NVIDIA driver did not enumerate (e.g. a missing dependency
    # of libGLX_nvidia, or no render node) so the fallback is diagnosable in CI.
    if command -v vulkaninfo >/dev/null 2>&1; then
      echo "--- NVIDIA Vulkan ICD diagnostic ---"
      VK_LOADER_DEBUG=warn vulkaninfo --summary 2>&1 | head -40 || true
      echo "--- end diagnostic ---"
    fi
    unset VK_ICD_FILENAMES
  else
    echo "WARNING: no NVIDIA Vulkan driver library found; using SwiftShader."
  fi
  install_swiftshader
}

VULKAN_SDK_VERSION="1.4.321.1"

# The no-argument default installs SwiftShader so the existing CPU-runner CI is
# unchanged. Pass "real-gpu" to prefer a real system ICD when one is present.
case "${1:-swiftshader}" in
  real-gpu)
    # Do not download the LunarG SDK here: its prebuilt glslc cannot run on the
    # old-glibc CUDA image. glslc comes from conda-forge and the loader from the
    # system package manager instead.
    install_vulkan_loader
    install_glslc
    setup_real_gpu_icd
    ;;
  swiftshader | *)
    install_swiftshader
    install_vulkan_sdk "${VULKAN_SDK_VERSION}"
    ;;
esac
