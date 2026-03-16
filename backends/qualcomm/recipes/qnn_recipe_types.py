# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from executorch.export import RecipeType


QNN_BACKEND: str = "qnn"


class QNNRecipeType(RecipeType):
    """QNN-specific recipe types"""

    # FP16 precision recipe, accepts kwargs:
    # 1. soc_model
    # 2. skip_node_id_set
    # 3. skip_node_op_set
    # 4. skip_mutable_buffer

    FP16 = "qnn_fp16"

    @classmethod
    def get_backend_name(cls) -> str:
        return QNN_BACKEND
