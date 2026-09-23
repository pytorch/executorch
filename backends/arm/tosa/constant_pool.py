# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import tosa_serializer as ts


_ConstantKey = tuple[Any, tuple[int, ...], bytes | None]


def _constant_key(shape, dtype, values) -> _ConstantKey:
    if dtype == ts.DType.SHAPE:
        if len(shape) > 1:
            raise ValueError(f"CONST_SHAPE expects rank metadata, got {shape}")
        rank = 0 if len(shape) == 0 else shape[0]
        constant = ts.TosaSerializerShape("", rank, values)
    else:
        constant = ts.TosaSerializerTensor("", shape, dtype, values)

    data = None if constant.data is None else bytes(constant.data)
    return constant.dtype, tuple(constant.shape), data


class TosaSerializerWithConstantPool(ts.TosaSerializer):
    """Pool generated constants independently within each TOSA basic block."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Native tensor wrappers retain their serializer. Cache names to avoid
        # an ownership cycle that Python's garbage collector cannot release.
        self._block_pools: dict[Any, dict[_ConstantKey, str]] = {}

    def addConst(self, shape, dtype, vals=None, name=""):
        """Return a matching constant in the current block or add a new one."""
        block = self.currRegion.currBasicBlock
        pool = self._block_pools.setdefault(block, {})
        key = _constant_key(shape, dtype, vals)
        if key not in pool:
            constant = super().addConst(shape, dtype, vals, name)
            pool[key] = constant.name
            return constant

        # Resolve the cached name to the object expected by callers. TOSA stores
        # shape constants separately from tensor constants.
        cached_name = pool[key]
        if dtype == ts.DType.SHAPE:
            constant = block.getShapeByName(cached_name)
        else:
            constant = block.getTensorByName(cached_name)
        return constant

    def addUnpooledConst(self, shape, dtype, vals=None, name=""):
        """Add a constant without pooling so its requested name remains
        addressable.
        """
        return super().addConst(shape, dtype, vals, name)
