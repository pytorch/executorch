# Copyright 2026 Mwaniki Njagi.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch.nn as nn


def remove_dropout(model: nn.Module) -> nn.Module:
    """
    Replace every nn.Dropout module with nn.Identity().

    The top-level model must be in eval() mode.
    """

    if model.training:
        raise ValueError("remove_dropout() requires model.eval()")

    for name, child in list(model.named_children()):
        if isinstance(child, nn.Dropout):
            setattr(model, name, nn.Identity())
        else:
            remove_dropout_from_children(child)

    return model


def remove_dropout_from_children(module: nn.Module) -> None:
    """
    Recursively replace Dropout modules.

    Deliberately does NOT check module.training.
    The top-level model has already been checked.
    """

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Dropout):
            setattr(module, name, nn.Identity())
        else:
            remove_dropout_from_children(child)
