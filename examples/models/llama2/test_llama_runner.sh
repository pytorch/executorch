#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test Llama runner in examples/models/llama2/main.cpp
# 1. Export a llama-like model
# 2. Build llama runner binary
# 3. Run model with the llama runner binary with prompt
set -e
# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../../../.ci/scripts/utils.sh"

cmake_install_executorch_libraries() {
    echo "Installing libexecutorch.a, libextension_module.so, libportable_ops_lib.a"
    rm -rf cmake-out
    retry cmake -DBUCK2="$BUCK" \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Release \
        -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
        -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
        -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
        -Bcmake-out .
    cmake --build cmake-out -j9 --target install --config Release
}

cmake_build_llama_runner() {
    echo "Building llama runner"
    dir="examples/models/llama2"
    retry cmake -DBUCK2="$BUCK" \
        -DCMAKE_INSTALL_PREFIX=cmake-out \
        -DCMAKE_BUILD_TYPE=Release \
        -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
        -Bcmake-out/${dir} \
        ${dir}
    cmake --build cmake-out/${dir} -j9 --config Release

}

cleanup_files() {
    rm tokenizer.model
    rm tokenizer.bin
    rm params.json
    rm result.txt
    rm llama2.pte
    rm stories110M.pt
}

if [[ $1 == "cmake" ]];
then
    cmake_install_executorch_libraries
    cmake_build_llama_runner
    download_stories_model_artifacts
    # Create tokenizer.bin
    echo "Creating tokenizer.bin"
    $PYTHON_EXECUTABLE -m examples.models.llama2.tokenizer.tokenizer -t tokenizer.model -o tokenizer.bin
    echo "Created tokenizer.bin"
    # Export model
    echo "Exporting model"
    # $PYTHON_EXECUTABLE -m examples.models.llama2.export_llama -c stories110M.pt -p params.json
    echo "Exported model as llama2.pte"
    # Run llama runner
    NOW=$(date +"%H:%M:%S")
    echo "Starting to run llama runner at ${NOW}"
    cmake-out/examples/models/llama2/llama_main --model_path="llama2.pte" --tokenizer_path=tokenizer.bin --prompt="Once" --temperature=0 --seq_len=10 > result.txt
    NOW=$(date +"%H:%M:%S")
    echo "Finished at ${NOW}"
    # verify correctness
    EXPECTED_PREFIX="Once upon a time,"
    echo "Expected result prefix: ${EXPECTED_PREFIX}"
    echo "Actual result: "
    cat result.txt
    echo ""
    if grep -q "$EXPECTED_PREFIX" "result.txt"; then
        echo "Success"
        cleanup_files
    else
        echo "Failure"
        cleanup_files
        exit 1
    fi
fi
