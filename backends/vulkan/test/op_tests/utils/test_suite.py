# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

###################################
## Generic Test Suite definition ##
###################################


class TestSuite:
    def __init__(self, input_cases: List[Any]):
        self.input_cases: List[Any] = input_cases
        self.prepacked_args: List[str] = []
        self.requires_prepack: bool = False
        self.dtypes: List[str] = ["at::kFloat", "at::kHalf"]

        self.data_gen: str = "make_rand_tensor"
        self.data_range = (0, 1)

        self.arg_dtype = {}
        self.arg_data_gen_fn: Dict[str, str] = {}
        self.arg_data_range = {}

        self.atol: str = "1e-5"
        self.rtol: str = "1e-5"

        self.is_view_op: bool = False
        self.test_name_suffix: Optional[str] = None

    def supports_prepack(self):
        return len(self.prepacked_args) > 0


##################################
## Vulkan Test Suite Definition ##
##################################


@dataclass
class VkTestSuite(TestSuite):
    def __init__(self, input_cases: List[Any]):
        super().__init__(input_cases)
        self.storage_types: List[str] = ["utils::kTexture3D"]
        self.layouts: List[str] = ["utils::kChannelsPacked"]
        self.data_gen: str = "make_rand_tensor"
        self.force_io: bool = True
        self.arg_storage_types: Dict[str, str] = {}
        self.arg_memory_layouts: Dict[str, str] = {}
