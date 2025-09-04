#!/bin/zsh

set -eux

# Check that we're in the executorch directory
current_dir=$(pwd)
if [[ ! "$current_dir" =~ executorch$ ]]; then
    echo "Error: This script must be run from a directory ending in 'executorch'"
    echo "Current directory: $current_dir"
    exit 1
fi

# Function to configure and build main project
configure_and_build_main() {
    local android_args=""
    if [[ "$ANDROID_MODE" == "true" ]]; then
        cmake . \
        -DCMAKE_INSTALL_PREFIX=$CMAKE_OUT_DIR \
        -DEXECUTORCH_BUILD_VULKAN=ON \
        -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-28 \
        -DGLSLC_PATH=$(which glslc) \
        -B$CMAKE_OUT_DIR
    else
        cmake . \
        -DCMAKE_INSTALL_PREFIX=$CMAKE_OUT_DIR \
        -DEXECUTORCH_BUILD_VULKAN=ON \
        -DGLSLC_PATH=$(which glslc) \
        -B$CMAKE_OUT_DIR
    fi

    cmake --build $CMAKE_OUT_DIR -j16 --target install
    # -DCMAKE_CXX_FLAGS="-DVULKAN_DEBUG" \
}

# Function to build main project only
build_main() {
    cmake --build $CMAKE_OUT_DIR -j16 --target install
}

# Function to configure and build tests
configure_and_build_tests() {
    # Check if glslc is installed
    if ! command -v glslc >/dev/null 2>&1; then
        echo "Error: glslc is not installed or not found in PATH."
        exit 1
    fi

    local android_args=""
    if [[ "$ANDROID_MODE" == "true" ]]; then
        cmake backends/vulkan/test/custom_ops/ \
            -DCMAKE_INSTALL_PREFIX=$CMAKE_OUT_DIR \
            -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
            -DANDROID_ABI=arm64-v8a \
            -DANDROID_PLATFORM=android-28 \
            -DGLSLC_PATH=$(which glslc) \
            -B$CMAKE_OUT_DIR/backends/vulkan/test/custom_ops
    else
        cmake backends/vulkan/test/custom_ops/ \
            -DCMAKE_INSTALL_PREFIX=$CMAKE_OUT_DIR \
            -DCMAKE_BUILD_TYPE=Debug \
            -DGLSLC_PATH=$(which glslc) \
            -B$CMAKE_OUT_DIR/backends/vulkan/test/custom_ops
    fi

    cmake --build $CMAKE_OUT_DIR/backends/vulkan/test/custom_ops -j16 --target all

}

build_tests() {
    cmake --build $CMAKE_OUT_DIR/backends/vulkan/test/custom_ops -j16 --target all
}

# Function to rebuild both main and tests
rebuild_both() {
    build_main
    build_tests
}

# Function to clean and rebuild everything
clean_and_rebuild() {
    rm -rf $CMAKE_OUT_DIR
    configure_and_build_main
    configure_and_build_tests
}

# Function to execute binary if specified
execute_binary() {
    local binary_name="$1"
    if [[ -n "$binary_name" ]]; then
        local binary_path="$CMAKE_OUT_DIR/backends/vulkan/test/custom_ops/$binary_name"
        echo "Executing binary: $binary_path"

        if [[ "$ANDROID_MODE" == "true" ]]; then
            if [[ -f "$binary_path" ]]; then
                echo "Pushing binary to Android device..."
                adb push "$binary_path" /data/local/tmp/
                echo "Executing binary on Android device..."
                adb shell "cd /data/local/tmp && ./$binary_name"
            else
                echo "Error: Binary '$binary_path' not found"
                exit 1
            fi
        else
            if [[ -f "$binary_path" && -x "$binary_path" ]]; then
                "$binary_path"
            else
                echo "Error: Binary '$binary_path' not found or not executable"
                exit 1
            fi
        fi
    fi
}

# Parse command line arguments
BINARY_TO_EXECUTE=""
ANDROID_MODE=false
CMAKE_OUT_DIR="cmake-out"

# Check for --android flag and adjust arguments accordingly
if [[ "$1" == "--android" ]]; then
    ANDROID_MODE=true
    CMAKE_OUT_DIR="cmake-android-out"
    shift  # Remove --android from arguments
    echo "Android mode enabled. Using $CMAKE_OUT_DIR as build directory."
fi

case "${1:-}" in
    --rebuild|-r)
        echo "Rebuilding both main project and tests..."
        BINARY_TO_EXECUTE="${2:-}"
        rebuild_both
        execute_binary "$BINARY_TO_EXECUTE"
        ;;
    --rebuild1|-r1)
        echo "Rebuilding main project only..."
        BINARY_TO_EXECUTE="${2:-}"
        build_main
        execute_binary "$BINARY_TO_EXECUTE"
        ;;
    --rebuild2|-r2)
        echo "Rebuilding tests only..."
        BINARY_TO_EXECUTE="${2:-}"
        build_tests
        execute_binary "$BINARY_TO_EXECUTE"
        ;;
    --clean|-c)
        echo "WARNING: This will delete the entire $CMAKE_OUT_DIR directory and rebuild everything."
        echo -n "Are you sure you want to continue? (y/N): "
        read -r response
        if [[ "$response" =~ ^[Yy]$ ]]; then
            echo "Cleaning and rebuilding everything..."
            BINARY_TO_EXECUTE="${2:-}"
            clean_and_rebuild
            execute_binary "$BINARY_TO_EXECUTE"
        else
            echo "Clean operation cancelled."
            exit 0
        fi
        ;;
    "")
        echo "Running full configure and build..."
        configure_and_build_main
        configure_and_build_tests
        ;;
    *)
        # If first argument doesn't match any build option, treat it as binary name
        # and use default build behavior
        echo "Running full configure and build..."
        BINARY_TO_EXECUTE="$1"
        configure_and_build_main
        configure_and_build_tests
        execute_binary "$BINARY_TO_EXECUTE"
        ;;
esac
