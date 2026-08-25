# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Stamped into Program.version of every exported PTE file. Keep in sync with
# Program::kMaxSupportedSchemaVersion in //executorch/runtime/executor/program.h,
# which is the highest version the C++ runtime agrees to load.
#
# Bump this only for a change that an older runtime would misread if it went
# unnoticed: a semantic change to an existing field, or a new field the runtime
# must understand to execute correctly. Purely additive optional fields stay
# backward/forward compatible (see schema/README.md) and need no bump.
EXECUTORCH_SCHEMA_VERSION = 0
