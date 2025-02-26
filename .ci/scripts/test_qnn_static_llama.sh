#!/bin/bash
# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -exu

source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

export EXECUTORCH_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export QNN_SDK_ROOT=/tmp/qnn/2.28.0.241029
export LD_LIBRARY_PATH="${QNN_SDK_ROOT}/lib/x86_64-linux-clang"
export PYTHONPATH=".."
cp schema/program.fbs exir/_serialize/program.fbs
cp schema/scalar_type.fbs exir/_serialize/scalar_type.fbs
cp -f build-x86/backends/qualcomm/PyQnnManagerAdaptor.cpython-310-x86_64-linux-gnu.so backends/qualcomm/python
cp -f build-x86/backends/qualcomm/PyQnnWrapperAdaptor.cpython-310-x86_64-linux-gnu.so backends/qualcomm/python

if [[ -z "${PYTHON_EXECUTABLE:-}" ]]; then
  PYTHON_EXECUTABLE=python3
fi

which "${PYTHON_EXECUTABLE}"

# Although static llama CI does not require graphviz, it is required by test_qnn_delegate.py
pip install graphviz

# Download stories llama110m artifacts
download_stories_model_artifacts
echo "Creating tokenizer.bin"
$PYTHON_EXECUTABLE -m extension.llm.tokenizer.tokenizer -t tokenizer.model -o tokenizer.bin

set +e
# Compile only as weight sharing is not applicable on x86
$PYTHON_EXECUTABLE backends/qualcomm/tests/test_qnn_delegate.py -k TestExampleLLMScript.test_llama_stories_110m --model SM8650 --build_folder build-android/ --executorch_root . --artifact_dir . --llama_artifacts . --compile_only
exit_code1=$?

# Checks accuracy with weight sharing disabled since x86 does not support weight sharing.
$PYTHON_EXECUTABLE backends/qualcomm/tests/test_qnn_delegate.py -k TestExampleLLMScript.test_llama_stories_110m --model SM8650 --build_folder build-x86/ --executorch_root . --artifact_dir . --llama_artifacts . --enable_x86_64
exit_code2=$?

# Check the exit codes and print messages
if [ $exit_code1 -ne 0 ]; then
    echo "Static Llama compile only with weight sharing test failed. $exit_code1."
fi

if [ $exit_code2 -ne 0 ]; then
    echo "Static Llama accuracy test failed. $exit_code2."
fi

# Return failure if either program failed
if [ $exit_code1 -ne 0 ] || [ $exit_code2 -ne 0 ]; then
    exit 1
else
    exit 0
fi
set -e
