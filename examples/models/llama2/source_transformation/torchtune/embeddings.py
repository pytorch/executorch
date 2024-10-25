# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from torchtune.modules.model_fusion._fusion import FusionEmbedding


def _replace_fusion_embeddings_with_nn_embedding(module: torch.nn.Module) -> None:
    """
    Replace TorchTune's FusionEmbedding with nn.Embedding. This is because
    the FusionEmbedding is meant for efficient training and bears no
    effect on inference. This is better since we get to avoid some of the
    potentially missing torch ops in the FusionEmbedding such as
    masked_select and masked_scatter.
    """
    
    for name, child in module.named_children():
        if isinstance(child, FusionEmbedding):
            setattr(
                module,
                name,
                torch.nn.Embedding(
                    child.embedding.num_embeddings + child.fusion_embedding.num_embeddings,
                    child.dim,
                )
            )
        else:
            _replace_fusion_embeddings_with_nn_embedding(child)

def replace_fusion_embeddings_with_nn_embedding(module: torch.nn.Module) -> torch.nn.Module:
    logging.info("Replacing fusion embeddings with nn.embeddings.")
    _replace_fusion_embeddings_with_nn_embedding(module)
    return module
    
