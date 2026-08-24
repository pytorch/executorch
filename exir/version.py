# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Stamped into Program.version of every exported PTE file. Keep in sync with
# Program::kMaxSupportedSchemaVersion in //executorch/runtime/executor/program.h,
# which is the highest version the C++ runtime agrees to load.
EXECUTORCH_SCHEMA_VERSION = 0
