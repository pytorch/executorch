# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Any, Optional

import torch
import torch.testing._internal.common_dtype as common_dtype

from executorch.exir.dialects.edge.arg.type import ArgType


class ArgMode(Enum):
    ONES = 0
    RANDOM = 1


class BaseArg:
    def __init__(
        self,
        argtype,
        *,
        value=None,
        size=None,
        fill=None,
        dtype=None,
        nonzero=False,
        nonneg=False,
        bounded=False,
    ):
        self.type: ArgType = argtype

        self.value_given = value is not None
        self.size_given = size is not None
        self.fill_given = fill is not None
        self.dtype_given = dtype is not None

        self.value = value
        self.size = (2, 2) if size is None else tuple(size)
        self.fill = 1 if fill is None else fill
        self.dtype = torch.float if dtype is None else dtype

        self.nonzero = nonzero
        self.nonneg = nonneg
        self.bounded = bounded

        self._mode: ArgMode = ArgMode.ONES
        self._kw: bool = False
        self._out: bool = False

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, v):
        if not isinstance(v, ArgMode):
            raise ValueError("mode property should be type ArgMode")
        self._mode = v

    @property
    def kw(self):
        return self._kw

    @kw.setter
    def kw(self, v):
        if not isinstance(v, bool):
            raise ValueError("kw property should be boolean")
        self._kw = v

    @property
    def out(self):
        return self._out

    @out.setter
    def out(self, v):
        if not isinstance(v, bool):
            raise ValueError("out property should be boolean")
        self._out = v

    def get_random_tensor(self, size, dtype):
        size = tuple(size)
        if dtype == torch.bool:
            if self.nonzero:
                return torch.full(size, True, dtype=dtype)
            else:
                return torch.randint(low=0, high=2, size=size, dtype=dtype)

        if dtype in common_dtype.integral_types():
            high = 100
        elif dtype in common_dtype.floating_types():
            high = 800
        else:
            raise ValueError(f"Unsupported Dtype: {dtype}")

        if dtype == torch.uint8:
            if self.nonzero:
                return torch.randint(low=1, high=high, size=size, dtype=dtype)
            else:
                return torch.randint(low=0, high=high, size=size, dtype=dtype)

        t = torch.randint(low=-high, high=high, size=size, dtype=dtype)
        if self.nonzero:
            pos = torch.randint(low=1, high=high, size=size, dtype=dtype)
            t = torch.where(t == 0, pos, t)
        if self.nonneg or self.bounded:
            t = torch.abs(t)

        if dtype in common_dtype.integral_types():
            return t
        if dtype in common_dtype.floating_types():
            return t / 8

    def get_random_scalar(self, dtype):
        return self.get_random_tensor([], dtype).item()

    def get_converted_scalar(self, value, dtype):
        if dtype == torch.bool:
            return bool(value)
        elif dtype in common_dtype.integral_types():
            return int(value)
        elif dtype in common_dtype.floating_types():
            return float(value)
        else:
            raise ValueError(f"Unsupported Dtype: {dtype}")

    def get_scalar_val_with_dtype(self, dtype):
        if self.value_given:
            return self.get_converted_scalar(self.value, dtype)
        elif self._mode == ArgMode.RANDOM:
            return self.get_random_scalar(dtype)
        elif self._mode == ArgMode.ONES:
            return self.get_converted_scalar(1, dtype)
        else:
            raise ValueError(f"Unsupported Mode: {self._mode}")

    def get_tensor_val_with_dtype(self, dtype):
        if self.value_given:
            return torch.tensor(self.value, dtype=dtype)
        elif self.fill_given:
            return torch.full(self.size, self.fill, dtype=dtype)
        elif self._mode == ArgMode.RANDOM:
            return self.get_random_tensor(self.size, dtype=dtype)
        elif self._mode == ArgMode.ONES:
            return torch.full(self.size, 1, dtype=dtype)
        elif self.size_given:
            return torch.full(self.size, self.fill, dtype=dtype)
        else:
            raise ValueError(f"Unsupported Mode: {self._mode}")

    def get_val_with_dtype(self, dtype):
        if dtype is None:
            return None
        if self.type.is_scalar_type():
            return dtype
        elif self.type.is_scalar():
            return self.get_scalar_val_with_dtype(dtype)
        elif self.type.is_tensor():
            return self.get_tensor_val_with_dtype(dtype)
        elif self.type.is_tensor_list():
            if not self.value_given:
                return []
            return [x.get_val_with_dtype(dtype) for x in self.value]
        else:
            raise ValueError(f"Unsupported Type: {self.type}")

    def get_val(self):
        if self.type.has_dtype():
            return self.get_val_with_dtype(self.dtype)
        else:
            return self.value


class BaseKwarg(BaseArg):
    def __init__(self, argtype, argname, **kwargs):
        BaseArg.__init__(self, argtype, **kwargs)
        self.argname = argname
        self._kw = True

    @property
    def kw(self):
        return super().kw


class InArg(BaseArg):
    def __init__(self, *args, **kwargs):
        BaseArg.__init__(self, *args, **kwargs)
        self._out = False

    @property
    def out(self):
        return self._out


class InKwarg(BaseKwarg, InArg):
    def __init__(self, *args, **kwargs):
        BaseKwarg.__init__(self, *args, **kwargs)


class OutArg(BaseKwarg):
    def __init__(self, argtype, *, argname="out", fill=0, **kwargs):
        BaseKwarg.__init__(self, argtype, argname, fill=fill, **kwargs)
        self._out = True

    @property
    def out(self):
        return self._out


class Return(BaseKwarg):
    """Model for returns of operators"""

    RETURN_NAME_PREFIX = "__ret"

    def __init__(self, argtype, *, argname=RETURN_NAME_PREFIX, fill=0, **kwargs):
        BaseKwarg.__init__(self, argtype, argname=argname, fill=fill, **kwargs)

    def is_expected(self, result: Any) -> bool:
        """Check whether return value matches expectation.
        For Tensor, we only focus on whether the return Tensor has the same dtype as expected.
        """
        if isinstance(result, torch.Tensor):
            return result.dtype == self.dtype
        else:
            raise NotImplementedError(f"Not implemented for {type(result)}")

    def to_out(self, *, name: Optional[str] = None) -> OutArg:
        return OutArg(
            self.type,
            argname=name if name else self.argname,
            fill=self.fill,
            size=self.size,
            dtype=self.dtype,
            value=self.value,
        )
