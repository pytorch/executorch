#!/bin/bash

# Build script for multimodal runner example
# This assumes ExecutorTorch has been built in the build/ directory

set -e  # Exit on any error

echo "Building multimodal runner example..."

# Create build directory for the example
mkdir -p build_example
cd build_example

# Run CMake with the custom CMakeLists file
cmake -DCMAKE_BUILD_TYPE=Release -S .. -B . -C ../CMakeLists_multimodal_example.txt

# Build the example
make -j$(nproc)

echo "Build completed! Executable is at: build_example/run_multimodal_runner"
echo ""
echo "Usage:"
echo "  ./build_example/run_multimodal_runner <model.pte> <tokenizer_path>"
echo ""
echo "Example:"
echo "  ./build_example/run_multimodal_runner model.pte tokenizer.model"