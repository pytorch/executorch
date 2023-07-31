# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import subprocess
import tempfile

import libfb.py.fbpkg as fbpkg


def serialize_program_to_flatbuffer(input_file: str, output_file: str) -> None:
    """
    Runs the serialization script in //executorch/exir/scripts:serialize
    """
    fbpkg_dir = fbpkg.fetch("executorch", tempfile.mkdtemp())
    cmd = [
        f"{fbpkg_dir}/serialize_to_flatbuffer",
        "-i",
        input_file,
        "-o",
        output_file,
    ]
    subprocess.check_output(cmd)
