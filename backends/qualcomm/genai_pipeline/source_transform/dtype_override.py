# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Module transforms casting the decoder's dtype."""

from __future__ import annotations

from typing import Any, Optional


def apply_dtype_override(module: Any, *, dtype_override: Optional[str]) -> Any:
    """Cast the module to ``--dtype-override``, if one was requested."""
    if dtype_override is None:
        return module

    from executorch.extension.llm.export.builder import DType

    return module.to(DType[dtype_override].to_torch_dtype())
