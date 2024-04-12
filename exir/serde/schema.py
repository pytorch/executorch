# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Additional schema from torch._export.serde.schema that is edge specific

from dataclasses import dataclass
from typing import List

import torch._export.serde.schema as export_schema


@dataclass
class CompileSpec:
    key: str
    value: str


@dataclass
class LoweredBackendModule:
    backend_id: str
    processed_bytes: str
    compile_specs: List[CompileSpec]
    original_module: export_schema.ExportedProgram
    original_state_dict: str


# NOTE: Please update this value if any modifications are made to the schema
SCHEMA_VERSION = (1, 0)
