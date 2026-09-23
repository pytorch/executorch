# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

# This module has moved to executorch.backends.test.program_builder.
# This re-export exists for backward compatibility.
#
# Resolved on attribute access rather than at import, because the target lives in a test
# package and the wheel does not ship those. Importing it here would make this module, which
# does ship, fail to import in an installed wheel.

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from executorch.backends.test.program_builder import (  # noqa: F401
        IrMode,
        ProgramBuilder,
    )

__all__ = ["IrMode", "ProgramBuilder"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from executorch.backends.test import program_builder

        return getattr(program_builder, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
