# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import List, Tuple, Union

from executorch.exir.tensor import TensorSpec

# @manual=fbsource//third-party/pypi/typing-extensions:typing-extensions
from typing_extensions import TypeAlias

ScalarSpec: TypeAlias = Union[int, float]
LeafValueSpec: TypeAlias = Union[TensorSpec, ScalarSpec]
ValueSpec: TypeAlias = Union[LeafValueSpec, List["ValueSpec"], Tuple["ValueSpec", ...]]
