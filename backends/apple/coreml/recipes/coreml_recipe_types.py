# Copyright © 2025 Apple Inc. All rights reserved.
#
# Please refer to the license found in the LICENSE file in the root directory of the source tree.


from executorch.export import RecipeType


COREML_BACKEND: str = "coreml"


class CoreMLRecipeType(RecipeType):
    """CoreML-specific generic recipe types"""

    # FP32 generic recipe, defaults to values published by the CoreML backend and partitioner
    # Precision = FP32, Default compute_unit = All (can be overriden by kwargs)
    FP32 = "coreml_fp32"

    # FP16 generic recipe, defaults to values published by the CoreML backend and partitioner
    # Precision = FP32, Default compute_unit = All (can be overriden by kwargs)
    FP16 = "coreml_fp16"

    @classmethod
    def get_backend_name(cls) -> str:
        return COREML_BACKEND
