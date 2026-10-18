# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy

import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval


def _fuse_sequential(module: nn.Sequential) -> tuple[nn.Sequential, int]:
    """Fold adjacent Conv2d -> BatchNorm2d pairs inside one nn.Sequential."""
    children = list(module.named_children())
    fused = OrderedDict()
    count = 0
    index = 0
    while index < len(children):
        name, child = children[index]
        if (
            isinstance(child, nn.Conv2d)
            and index + 1 < len(children)
            and isinstance(children[index + 1][1], nn.BatchNorm2d)
        ):
            bn_name, batch_norm = children[index + 1]
            if child.training or batch_norm.training:
                raise RuntimeError(
                    f"{name} / {bn_name} must be in eval() mode to fold batch norm"
                )
            fused[name] = fuse_conv_bn_eval(child, batch_norm)
            count += 1
            index += 2
            continue
        fused[name] = child
        index += 1
    return nn.Sequential(fused), count


def _fuse_recursive(module: nn.Module) -> int:
    total = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Sequential):
            fused_child, count = _fuse_sequential(child)
            setattr(module, name, fused_child)
            total += count + _fuse_recursive(fused_child)
        else:
            total += _fuse_recursive(child)
    return total


def fold_batch_norm(
    model: nn.Module, *, inplace: bool = False
) -> tuple[nn.Module, int]:
    """
    Fold inference-time Conv2d -> BatchNorm2d pairs held in nn.Sequential
    containers (the torchvision convention) before export, so that the
    quantizer sees plain convolutions. Returns (model, number_of_folds).
    """
    if model.training:
        raise ValueError("fold_batch_norm() requires model.eval()")
    if not inplace:
        model = deepcopy(model)
    return model, _fuse_recursive(model)


def count_conv_bn(model: nn.Module) -> tuple[int, int]:
    """Return (number of Conv2d modules, number of BatchNorm2d modules)."""
    convs = sum(isinstance(m, nn.Conv2d) for m in model.modules())
    norms = sum(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    return convs, norms


__all__ = ["count_conv_bn", "fold_batch_norm"]
