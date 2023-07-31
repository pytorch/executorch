#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# Build python docs and move to static/py_api
buck2 run //executorch/docs:sphinx-build -- -M html ../source_py/ sphinxbuild_py

rm -r static/py_api || true
mv sphinxbuild_py/html/ static/py_api
rm -r sphinxbuild_py

# Build C++ docs and move to static/cpp_api
buck2 run //executorch/docs:sphinx-build -- -M html ../source_cpp/ sphinxbuild_cpp

rm -r static/cpp_api || true
mv sphinxbuild_cpp/html/ static/cpp_api
rm -r sphinxbuild_cpp
