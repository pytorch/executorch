# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2024-25 Arm Limited and/or its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash

set -e

echo "Building executor_runner with profiling enabled..."

cmake --preset profiling
cmake --build cmake-out --target executor_runner

echo "Build completed successfully!"

MODEL_PATH=${1:-"my_model"}
ETDUMP_PATH=${2:-"path_to_et_dump"}

echo "Running and profiling model: $MODEL_PATH"
echo "ETDump output path: $ETDUMP_PATH"

./cmake-out/executor_runner --model_path="$MODEL_PATH" --etdump_path="$ETDUMP_PATH"

echo "Profiling run completed!"

echo "Generating profiling CSV..."
python devtools/scripts/generate_profiling_csv.py --etdump_path="$ETDUMP_PATH" --output="op_profiling.csv"

echo "Profiling CSV generated: op_profiling.csv"
echo "Profiling workflow completed successfully!"
