# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-25 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

# ExecutorTorch Model Profiling Script
#
# This script automates the process of building executor_runner with profiling enabled,
# running model inference with ETDump collection, and generating CSV profiling reports.
#
# Usage:
#   ./devtools/scripts/profile_model.sh [model_path] [etdump_path]
#
# Arguments:
#   model_path  - Path to the .pte model file (default: "my_model")
#   etdump_path - Path for ETDump output file (default: "path_to_et_dump")
#
# Examples:
#   ./devtools/scripts/profile_model.sh
#   ./devtools/scripts/profile_model.sh llama3.pte llama3_etdump
#
# Note: This script must be run from the top-level executorch directory.

set -e

echo "Building executor_runner with profiling enabled..."

cmake --preset profiling -B build-profiling -DCMAKE_BUILD_TYPE=Release
cmake --build build-profiling --target executor_runner

echo "Build completed successfully!"

MODEL_PATH=${1:-"my_model"}
ETDUMP_PATH=${2:-"path_to_et_dump"}

echo "Running and profiling model: $MODEL_PATH"
echo "ETDump output path: $ETDUMP_PATH"

./build-profiling/executor_runner --model_path="$MODEL_PATH" --etdump_path="$ETDUMP_PATH"

echo "Profiling run completed!"

echo "Generating profiling CSV..."
python devtools/scripts/generate_profiling_csv.py --etdump_path="$ETDUMP_PATH" --output="op_profiling.csv"

echo "Profiling CSV generated: op_profiling.csv"
echo "Profiling workflow completed successfully!"
