#!/bin/bash

set -e

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_DIR=$(realpath ${BASE_DIR}/../../)

echo PROJECT_DIR=${PROJECT_DIR}

ANDROID_ABI=arm64-v8a
ANDROID_PLATFORM=android-28 # Trace requires over android-23

echo ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT}
echo ANDROID_ABI=${ANDROID_ABI}
echo ANDROID_PLATFORM=${ANDROID_PLATFORM}

main() {
      cd "$PROJECT_DIR"
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
            -B"${build_dir_example}" \
            "$PROJECT_DIR/${example_root}"
      cmake --build build_samsung_android/examples/samsung/ --config Release
}

main "$@"
