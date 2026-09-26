# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Llama2Model

__all__ = ["Llama2Model"]


def __getattr__(name):
    if name == "Llama2Model":
        from .model import Llama2Model

        return Llama2Model
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
