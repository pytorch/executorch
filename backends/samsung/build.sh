#!/bin/bash
## Directory Info
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
PROJECT_DIR=$(realpath ${BASE_DIR}/../../)

X86_64_BUILD_DIR=${PROJECT_DIR}/build_samsung
ANDROID_BUILD_DIR=${PROJECT_DIR}/build_samsung_android
BUILD_ARCH=all
CLEAN_BUILD_DIR=false

EXYNOS_AI_LITECORE_ROOT=${EXYNOS_AI_LITECORE_ROOT}
ANDROID_NDK_ROOT=${ANDROID_NDK_ROOT}

function usage() {
  echo "Usage build.sh <command> <arguments>

    <command>           <argument>                 <description>

    --sdk                                      The path of downloaded ENN SDK, which is required for building.
                                               Or export EXYNOS_AI_LITECORE_ROOT=/path/to/xxx
    --ndk                                      The path of Android NDK, or export ANDROID_NDK_ROOT=/path/to/ndk.

    --build, -b     [x86_64, android, all]     Default is all, x86_64 target to offline compilation,
                                               android target to online execution.
    --clean, -c                                Clean the build cache.
    --help, -h                                 Print the usage information.
  "
}

function build_x86_64() {
  if [[ -z ${EXYNOS_AI_LITECORE_ROOT} ]]; then
    echo "Please export EXYNOS_AI_LITECORE_ROOT or set by command"
    exit 1
  fi

  echo "EXYNOS_AI_LITECORE_ROOT: ${EXYNOS_AI_LITECORE_ROOT}"
  echo "ANDROID_NDK_ROOT: ${ANDROID_NDK_ROOT}"

  cmake \
        -DCMAKE_INSTALL_PREFIX=${X86_64_BUILD_DIR} \
        -DEXYNOS_AI_LITECORE_ROOT=${EXYNOS_AI_LITECORE_ROOT} \
        -DEXECUTORCH_BUILD_ENN=ON \
        -DEXECUTORCH_BUILD_DEVTOOLS=ON \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
	      -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
        -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
        -S ${PROJECT_DIR} \
        -B ${X86_64_BUILD_DIR}

  cmake --build ${X86_64_BUILD_DIR} -j $(nproc) --target install

  rm -f ${PROJECT_DIR}/backends/samsung/python/Py*so
  cp -fv ${X86_64_BUILD_DIR}/backends/samsung/Py*so ${PROJECT_DIR}/backends/samsung/python/
  cp -fv ${PROJECT_DIR}/schema/*.fbs ${PROJECT_DIR}/exir/_serialize/
}

function build_android() {
  if [[ -z ${ANDROID_NDK_ROOT} ]]; then
    echo "Please export ANDROID_NDK_ROOT or set by command"
    exit 1
  fi

  ANDROID_ABI=arm64-v8a
  ANDROID_PLATFORM=android-28 # Trace requires over android-23

  cmake \
        -DCMAKE_INSTALL_PREFIX=${ANDROID_BUILD_DIR} \
        -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK_ROOT}/build/cmake/android.toolchain.cmake" \
        -DANDROID_NDK=${ANDROID_NDK} \
        -DANDROID_ABI="${ANDROID_ABI}" \
        -DANDROID_PLATFORM=${ANDROID_PLATFORM} \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_ENN=ON \
        -DEXYNOS_AI_LITECORE_ROOT=${EXYNOS_AI_LITECORE_ROOT} \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
	      -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
        -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
        -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
        -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON \
        -DEXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL=ON \
        -DEXECUTORCH_ENABLE_LOGGING=1 \
        -DEXECUTORCH_BUILD_DEVTOOLS=ON \
        -DEXECUTORCH_ENABLE_EVENT_TRACER=ON \
        -S ${PROJECT_DIR} \
        -B ${ANDROID_BUILD_DIR}

  cmake --build ${ANDROID_BUILD_DIR} -j $(nproc) --target install
}

# Main
for (( i=1; i<=$#; i++))
do
    case "${!i}" in
    "--sdk")
      let i++
      EXYNOS_AI_LITECORE_ROOT="${!i}"
      ;;
    "--ndk")
      let i++
      ANDROID_NDK_ROOT="${!i}"
      ;;
    "--clean"|"-c")
      CLEAN_BUILD_DIR=true
      ;;
    "--build"|"-b")
      let i++
      BUILD_ARCH="${!i}"
      ;;
    "--help"|"-h")
      usage
      exit 0
      ;;
    *)
    echo "Unknown option: ${!i}"
      usage
      exit 1
      ;;
  esac
done

cd ${PROJECT_DIR}
if [ "${CLEAN_BUILD_DIR}" = true ]; then
  rm -rf ${X86_64_BUILD_DIR}
  rm -rf ${ANDROID_BUILD_DIR}
  exit 0
fi

if [[ "${BUILD_ARCH}" = "all" || "${BUILD_ARCH}" = "x86_64" ]]; then
  build_x86_64
fi
if [[ "${BUILD_ARCH}" = "all" || "${BUILD_ARCH}" = "android" ]]; then
  build_android
fi
