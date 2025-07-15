# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable

    # This only runs during static type checking (not at runtime)
    def expand(arg: object) -> Callable[..., None]: ...

else:
    # Real import used at runtime
    # from parameterized.parameterized import parameterized.expand as expand # noqa
    from parameterized.parameterized import parameterized

    expand = parameterized.expand
