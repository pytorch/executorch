# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

# This module has moved to executorch.backends.test.graph_builder.
# This re-export exists for backward compatibility.
#
# Resolved on attribute access rather than at import, because the target lives in a test
# package and the wheel does not ship those. Importing it here would make this module, which
# does ship, fail to import in an installed wheel.

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from executorch.backends.test.graph_builder import (  # noqa: F401
        GraphBuilder,
        single_op_builder,
    )

__all__ = ["GraphBuilder", "single_op_builder"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from executorch.backends.test import graph_builder

        return getattr(graph_builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
