# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from executorch.export import RecipeType


class AOTIRecipeType(RecipeType):
    """AOTInductor-specific recipe types"""

    FP32 = "fp32"
    # more to be added...

    @classmethod
    def get_backend_name(cls) -> str:
        return "aoti"
