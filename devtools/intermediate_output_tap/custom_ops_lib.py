# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

from __future__ import annotations

from torch.library import impl, Library

lib: Library = Library("executorch_devtools", "DEF")

lib.define("tap.Tensor(Tensor x, str reducer_name) -> Tensor")


@impl(lib, "tap.Tensor", "CompositeExplicitAutograd")
def tap_tensor_impl(x, reducer_name):
    # Defer the import to break a module-import cycle (`_reducers` → torch →
    # custom_ops_lib registration).
    from executorch.devtools.intermediate_output_tap._reducers import get_reducer

    return get_reducer(reducer_name).eager(x)
