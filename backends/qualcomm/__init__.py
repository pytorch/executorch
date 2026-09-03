# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Deliberately empty. Importing this package must have no side effects, and it must not be the
# home of anything a submodule needs, because build systems that assemble a package from a file
# list can leave this file out and synthesize an empty one in its place. The Qualcomm SDK setup
# that used to live here is in utils/qnn_sdk_setup.py, called by the code paths that start a
# backend.
