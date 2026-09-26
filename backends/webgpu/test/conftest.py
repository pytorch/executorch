# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Workaround for PyTorch 2.11 bug where LeafSpec dataclass fields
# (type, _context, _children) are not initialized by the C++ constructor,
# causing AttributeError in run_decompositions and copy.deepcopy.
import dataclasses

from torch.utils._pytree import LeafSpec


def _leafspec_getattr(self, name):  # type: ignore[no-untyped-def]
    for f in dataclasses.fields(type(self)):
        if f.name == name:
            if f.default is not dataclasses.MISSING:
                return f.default
            elif f.default_factory is not dataclasses.MISSING:
                val = f.default_factory()
                object.__setattr__(self, name, val)
                return val
    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


if not hasattr(LeafSpec(), "type"):
    LeafSpec.__getattr__ = _leafspec_getattr
