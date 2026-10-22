# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.export import RecipeType


COREAI_BACKEND: str = "coreai"


class CoreAIRecipeType(RecipeType):
    """Core AI-specific recipe types."""

    # FP32 precision: lower as-is to Core AI.
    FP32 = "coreai_fp32"

    # FP16 precision: cast the exported program to FP16 (via coreai_opt's
    # cast_fp32_to_fp16) before edge lowering, then lower to Core AI.  Note this
    # makes the model's external I/O FP16.
    FP16 = "coreai_fp16"

    @classmethod
    def get_backend_name(cls) -> str:
        return COREAI_BACKEND
