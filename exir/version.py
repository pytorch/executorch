# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# Stamped into Program.version of an exported PTE file that uses nothing that
# needs a newer version, so every runtime loads it.
EXECUTORCH_SCHEMA_VERSION = 0

# The highest version stamped into Program.version of an exported PTE file.
# Keep in sync with Program::kMaxSupportedSchemaVersion in
# //executorch/runtime/executor/program.h, which is the highest version the C++
# runtime agrees to load. Each file gets the highest of the versions below
# among the features it uses, so older runtimes keep loading files that need
# nothing newer.
#
# Bump this only for a change that an older runtime would misread if it went
# unnoticed: a semantic change to an existing field, or a new field the runtime
# must understand to execute correctly. Purely additive optional fields stay
# backward/forward compatible (see schema/README.md) and need no bump.
MAX_EXECUTORCH_SCHEMA_VERSION = 1

# The version that each such change needs. The exporter takes a file's version
# from the features it uses, so a later bump of MAX_EXECUTORCH_SCHEMA_VERSION
# leaves other files at the version they had.
#
# Methods share a planned buffer (ExecutionPlan.shared_non_const_buffers).
# Runtimes that ignore the field would run some of these programs wrongly: a
# method that does not share the buffer but keeps host data of its own in the
# bytes of the shared state would overwrite it. Shared host state that those
# runtimes already share, with share_memory_arenas, stays at version 0, so they
# keep loading it, with 2 differences: loading a method late resets the
# state to its initial data, and a method that stores no initial data reads
# zeros if no method that stores it has loaded yet.
# Runtimes before 1.5.0 do not check the version, so they also load version 1
# files and run them wrongly; 1.5.0 to 1.6 refuse them. A runtime that accepts
# version 1 still relies on its runner to give every method that marks a buffer
# one allocation of it, as Module and the Python runtime do. Program and Method
# do not read the field.
SHARED_NON_CONST_BUFFERS_VERSION = 1
