# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Callable, List, Optional, Union

import torch
import torch.fx.passes.infra.pass_manager as fx
import torch.utils._pytree as pytree
from executorch.exir.error import ExportError, ExportErrorType
from torch.fx.passes.infra.pass_base import PassResult
from typing_extensions import TypeAlias

PassType: TypeAlias = Callable[[torch.fx.GraphModule], Optional[PassResult]]


class PassManager(fx.PassManager):
    """
    Class to run multiple passes on a given graph module. The PassManager is
    callable so to run it, we can just call the PassManager instance.

    Private Attributes:
        * **passes**: A list of callable passes
        * **params**: An instance of PassManagerParams containing the result of the
            flags set in the constructor.
    """

    def __init__(
        self,
        passes: Optional[Union[List[PassType], List[List[PassType]]]] = None,
        run_checks_after_each_pass: bool = False,
        suppress_check_failures: bool = False,
    ) -> None:
        r"""
        Args:
            passes: A list of passes
            enable_debug_pass: set to true to enable the debug passes
            run_checks_after_each_pass: whether to run checks and linting after each pass
        """

        # Flatten the passes to a list of callables
        passes = passes if passes else []
        flattened_passes = [
            fx.pass_result_wrapper(fn) for fn in pytree.tree_flatten(passes)[0]
        ]

        super().__init__(
            flattened_passes,
            run_checks_after_each_pass=run_checks_after_each_pass,
            suppress_check_failures=suppress_check_failures,
        )

    def check(self, module: torch.nn.Module) -> None:
        """
        Runs various checks on the given graph module to make sure it contains
        the needed data for passes.

        Some checks that need to be run:
            - Ensure that types of operator node match the types specified in
              the node's spec field (ex. if the op returns a tuple then the
              node's spec field is a tuple)
            - Ensure that the graph module has type torch.fx.GraphModule
        """
        assert isinstance(module, fx.GraphModule)
        module.recompile()
        module.graph.lint()
        # TODO(qihan): use verifier.check_is_exir

        for node in module.graph.nodes:
            if node.op == "call_method":
                raise ExportError(
                    ExportErrorType.NOT_SUPPORTED,
                    f"call_method `{node}` is not supported except for backend delegate.",
                )
