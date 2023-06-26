# pyre-strict

from typing import List, Tuple, Union

from executorch.exir.tensor import TensorSpec

# @manual=fbsource//third-party/pypi/typing-extensions:typing-extensions
from typing_extensions import TypeAlias

ScalarSpec: TypeAlias = Union[int, float]
LeafValueSpec: TypeAlias = Union[TensorSpec, ScalarSpec]
ValueSpec: TypeAlias = Union[LeafValueSpec, List["ValueSpec"], Tuple["ValueSpec", ...]]
