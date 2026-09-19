# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared constants for the Core AI backend compiler."""

# The single entrypoint / graph name of a per-delegate coreai AIProgram.  Each
# ExecuTorch delegate is converted to its own coreai program whose only graph is
# named "main" (``save_asset`` emits ``main.hash`` / ``main.mlirb`` accordingly).
MAIN_ENTRYPOINT = "main"
