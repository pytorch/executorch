# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.export import RecipeType


class CPURecipeType(RecipeType):
    FP32 = "cpu_fp32"

    @classmethod
    def get_backend_name(cls) -> str:
        return "cpu"
