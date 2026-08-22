# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import re
import warnings
from typing import Callable, Dict, List, Optional

from executorch.exir.error import ExportError, ExportErrorType

from executorch.exir.pass_manager import PassType


class PassRegistry:
    """
    Allows passes to be automatically registered into a global registry, and
    users to search within the registry by the pass’s string name to get a pass.

    Attributes:
        registry: A dictionary of names of passes mapping to a list of passes in
        the form of callable functions or PassBase instances (which are also callable)
    """

    registry: Dict[str, List[PassType]] = {}

    @classmethod
    def register(
        cls, pass_name: Optional[str] = None
    ) -> Callable[[PassType], PassType]:
        """
        A decorator used on top of passes to insert a pass into the registry. If
        pass_name is not specified, then it will be generated based on the name
        of the function passed in.

        This decorator can be used on top of functions (with type
        PassManagerParams * torch.fx.GraphModule -> None) or on top of PassBase
        subclasses instances.
        """

        def wrapper(one_pass: PassType) -> PassType:
            key = pass_name
            if not key:
                key = re.sub(r"(?<!^)(?=[A-Z])", "_", one_pass.__name__).lower()

            cls.register_list(key, [one_pass])
            return one_pass

        return wrapper

    @classmethod
    def register_list(cls, pass_name: str, pass_list: List[PassType]) -> None:
        """
        A function used to insert a list of passes into the registry. The pass
        can be searched for in the registry according to the given pass name.
        """

        if pass_name in cls.registry:
            warnings.warn(
                f"Pass {pass_name} already exists inside of the PassRegistry. Will ignore.",
                stacklevel=1,
            )
            return

        cls.registry[pass_name] = pass_list

    @classmethod
    def get(cls, key: str) -> List[PassType]:
        """
        Gets the pass corresponding to the given name. If the pass is a function
        then it will directly return the callable function.

        Args:
            key: The name of a pass

        Return:
            A callable pass or a list of callable passes
        """
        if key not in cls.registry:
            raise ExportError(
                ExportErrorType.MISSING_PROPERTY,
                f"Pass {key} does not exists inside of the PassRegistry",
            )

        pass_found = cls.registry[key]
        return pass_found
