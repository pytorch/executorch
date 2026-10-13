#!/bin/bash

set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_DIR=$(realpath ${BASE_DIR}/../../)

echo PROJECT_DIR=${PROJECT_DIR}

if [[ -z ${ANDROID_NDK_ROOT} ]]; then
  echo "Please export ANDROID_NDK_ROOT"
  exit 1
fi

ANDROID_ABI=arm64-v8a
ANDROID_PLATFORM=android-28 # Trace requires over android-23

echo ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT}
echo ANDROID_ABI=${ANDROID_ABI}
echo ANDROID_PLATFORM=${ANDROID_PLATFORM}

usage() {
      echo "Usage: $0 [--llm-wrapper-so]"
      echo "    --llm-wrapper-so    Also build libexynos_llm_wrapper.so and exynos_llm_wrapper_sample."
}

main() {
      local build_llm_wrapper_so=OFF
      for arg in "$@"; do
          case "$arg" in
          "--llm-wrapper-so") build_llm_wrapper_so=ON ;;
          "--help"|"-h") usage; exit 0 ;;
          *) echo "Unknown option: $arg"; usage; exit 1 ;;
          esac
      done

      cd "$PROJECT_DIR"

      if [[ -z ${ANDROID_NDK_ROOT} ]]; then
          echo "Please export ANDROID_NDK_ROOT by setting andrid NDK path"
          echo "$ export ANDROID_NDK_ROOT = path of android NDK"
          exit 1
      fi

      if [[ -z ${ANDROID_NDK} ]]; then
          ANDROID_NDK=${ANDROID_NDK_ROOT}
      fi

      local build_dir_root="build_samsung_android"
      local example_root="examples/samsung"
      local build_dir_example="$PROJECT_DIR/${build_dir_root}/${example_root}"
      local cmake_prefix_path="$PROJECT_DIR/${build_dir_root}/lib/cmake/ExecuTorch;$PROJECT_DIR/${build_dir_root}/third-party/gflags;$PROJECT_DIR/${build_dir_root}/lib/cmake/tokenizers;$PROJECT_DIR/${build_dir_root}/lib/cmake/re2;$PROJECT_DIR/${build_dir_root}/lib/cmake/absl;"

      echo build_dir=${build_dir_root}
      echo build_dir_example=${build_dir_example}
      echo cmake_prefix_path=${cmake_prefix_path}

      cmake -DCMAKE_PREFIX_PATH=${cmake_prefix_path} \
            -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK_ROOT/build/cmake/android.toolchain.cmake" \
            -DANDROID_NDK=$ANDROID_NDK \
            -DANDROID_ABI="$ANDROID_ABI" \
            -DANDROID_PLATFORM=$ANDROID_PLATFORM \
            -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH \
            -DCMAKE_BUILD_TYPE=Release \
            -DEXYNOS_BUILD_LLM_WRAPPER_SO=${build_llm_wrapper_so} \
            -B"${build_dir_example}" \
            "$PROJECT_DIR/${example_root}"
      cmake --build ${build_dir_example} --config Release

      if [[ ${build_llm_wrapper_so} == OFF ]]; then
          rm -f "${build_dir_example}/libexynos_llm_wrapper.so" \
                "${build_dir_example}/exynos_llm_wrapper_sample"
      fi
}

main "$@"
