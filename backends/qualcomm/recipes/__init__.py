# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""QNN Recipe module for ExecuTorch"""
from executorch.export import recipe_registry

from .qnn_recipe_provider import QNNRecipeProvider
from .qnn_recipe_types import QNNRecipeType

# Auto-register XNNPACK recipe provider
recipe_registry.register_backend_recipe_provider(QNNRecipeProvider())

__all__ = ["QNNRecipeProvider", "QNNRecipeType"]
