# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

# selective_build links the shipped runtime DLL in executorch/lib, and Windows
# records no search path in a DLL. Not resolved: in an editable install this
# directory is a symlink, and executorch/lib sits beside the link.
_lib_dir = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, "lib"
    )
)
if sys.platform == "win32" and os.path.isdir(_lib_dir):
    os.add_dll_directory(_lib_dir)
del os, sys, _lib_dir
