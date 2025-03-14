# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# @lint-ignore-every LICENSELINT
# type: ignore[syntax]
from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    version="0.1.0",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(where="pytorch_tokenizers"),
    package_dir={"": "pytorch_tokenizers"},
)
