# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Decorators used to warn about non-stable APIs."""

# pyre-strict

from typing import Any, Dict, Optional, Sequence, Type

from typing_extensions import deprecated

__all__ = ["deprecated", "experimental"]


class ExperimentalWarning(DeprecationWarning):
    """Emitted when calling an experimental API.

    Derives from DeprecationWarning so that it is similarly filtered out by
    default.
    """

    def __init__(self, /, *args: Sequence[Any], **kwargs: Dict[str, Any]) -> None:
        super().__init__(*args, **kwargs)


class experimental(deprecated):
    """Indicates that a class, function or overload is experimental.

    When this decorator is applied to an object, the type checker
    will generate a diagnostic on usage of the experimental object.
    """

    def __init__(
        self,
        message: str,
        /,
        *,
        category: Optional[Type[Warning]] = ExperimentalWarning,
        stacklevel: int = 1,
    ) -> None:
        super().__init__(message, category=category, stacklevel=stacklevel)
