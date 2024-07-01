# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass


@dataclass
class RuntimeInfo:
    key: str  # runtime info key like "runtime_version"
    value: bytes  # runtime info value like "v0.4.2"
