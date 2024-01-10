# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test helper for exporting an nn.Module to an ExecuTorch program."""

import functools
import inspect
from typing import Callable, Sequence, Type

import executorch.exir as exir
from executorch.exir import ExecutorchBackendConfig, ExecutorchProgram
from executorch.exir.dynamic_shape import DynamicMemoryPlanningMode
from executorch.exir.pass_manager import PassManager
from executorch.exir.passes import (
    DebugPass,
    MemoryPlanningPass,
    to_scratch_op_pass,
    ToOutVarPass,
)
from torch import nn
from torch.fx import GraphModule


class ExportedModule:
    """The result of exporting an nn.Module.

    Attributes:
        eager_module: The original nn.Module that was exported.
        methods: The names of the eager_module methods that were traced.
        executorch_program: The resulting ExecutorchProgram.
        graph_module: The resulting GraphModule.
        trace_inputs: The inputs that were used when tracing eager_module.
    """

    def __init__(
        self,
        eager_module: nn.Module,
        methods: Sequence[str],
        executorch_program: ExecutorchProgram,
        graph_module: GraphModule,
        trace_inputs: Sequence,
        get_random_inputs_fn: Callable[[], Sequence],
    ):
        """INTERNAL ONLY: Use ExportedModule.export() instead."""
        self.eager_module: nn.Module = eager_module
        self.methods: Sequence[str] = methods
        self.executorch_program: ExecutorchProgram = executorch_program
        self.graph_module: GraphModule = graph_module
        self.trace_inputs: Sequence = trace_inputs
        self.__get_random_inputs_fn = get_random_inputs_fn

    def get_random_inputs(self) -> Sequence:
        """Returns random inputs appropriate for model inference."""
        return self.__get_random_inputs_fn()

    @staticmethod
    def export(
        module_class: Type[nn.Module],
        methods: Sequence[str] = ("forward",),
        ignore_to_out_var_failure: bool = False,
        dynamic_memory_planning_mode: DynamicMemoryPlanningMode = DynamicMemoryPlanningMode.UPPER_BOUND,
        capture_config=None,
        extract_constant_segment: bool = True,
    ) -> "ExportedModule":
        """
        Creates a new ExportedModule for the specified module class.

        Args:
            module_class: The subclass of nn.Module to export.
            methods: The names of the module_class methods to trace.
            ignore_to_out_var_failure: Whether to ignore the failue when an
                functional op does not have an out variant.
            dynamic_memory_planning_mode: The dynamic memory planning mode to
                use.
        """

        def get_inputs_adapter(
            worker_fn: Callable, method: str
        ) -> Callable[[], Sequence]:
            """Returns a function that may bind `method` as a parameter of
            `worker_fn`, and ensures that `worker_fn` always returns a list or
            tuple.

            Args:
                worker_fn: The function to wrap. Must take zero or one
                    arguments. If it takes one argument, that argument must be
                    called "method" and expect a string.
                method: The name of the method to possibly pass to `worker_fn`.

            Returns:
                A function that takes zero arguments and returns a Sequence.
            """
            # Names of the parameters of worker_fn.
            params = inspect.signature(worker_fn).parameters.keys()
            if len(params) == 1:
                assert "method" in params, f"Expected 'method' param in {params}"
                # Bind our `method` parameter to `worker_fn`, which has the
                # signature `func(method: str)`.
                worker_fn = functools.partial(worker_fn, method)
            else:
                assert len(params) == 0, f"Unexpected params in {params}"
                # worker_fn takes no parameters.

            def return_wrapper():
                inputs = worker_fn()
                # Wrap the return value in a tuple if it's not already a tuple
                # or list.
                if not isinstance(inputs, (tuple, list)):
                    inputs = (inputs,)
                return inputs

            return return_wrapper

        # Create the eager module.
        eager_module = module_class().eval()

        # Generate inputs to use while tracing.
        trace_inputs_method = "get_upper_bound_inputs"
        get_trace_inputs = get_inputs_adapter(
            getattr(eager_module, trace_inputs_method)
            if hasattr(eager_module, trace_inputs_method)
            else eager_module.get_random_inputs,
            # all exported methods must have the same signature so just pick the first one.
            methods[0],
        )
        trace_inputs: Sequence = get_trace_inputs()
        method_name_to_args = {}
        for method in methods:
            method_name_to_args[method] = trace_inputs

        method_name_to_constraints = None
        if hasattr(eager_module, "get_constraints"):
            assert capture_config is not None
            assert capture_config.enable_aot is True
            trace_constraints = eager_module.get_constraints()
            method_name_to_constraints = {}
            for method in methods:
                method_name_to_constraints[method] = trace_constraints

        memory_planning_pass = MemoryPlanningPass("greedy")
        if hasattr(eager_module, "get_memory_planning_pass"):
            memory_planning_pass = eager_module.get_memory_planning_pass()

        # Capture an executorch program.
        executorch_program = (
            exir.capture_multiple(
                eager_module,
                method_name_to_args,
                capture_config,
                constraints=method_name_to_constraints,
            )
            .to_edge(exir.EdgeCompileConfig(_check_ir_validity=False))
            .to_executorch(
                ExecutorchBackendConfig(
                    passes=[
                        DebugPass(
                            show_src=True,
                            show_spec=False,
                            show_full_path=True,
                            show_all_frames=True,
                        ),
                        to_scratch_op_pass,
                    ],
                    dynamic_memory_planning_mode=dynamic_memory_planning_mode,
                    memory_planning_pass=memory_planning_pass,
                    to_out_var_pass=ToOutVarPass(ignore_to_out_var_failure),
                    extract_constant_segment=extract_constant_segment,
                )
            )
        )

        # Generate the graph module created during capture.
        graph_module = executorch_program.dump_graph_module()
        graph_module = PassManager(
            passes=[DebugPass(show_spec=True)], run_checks_after_each_pass=True
        )(graph_module).graph_module

        # Get a function that creates random inputs appropriate for testing.
        get_random_inputs_fn = get_inputs_adapter(
            eager_module.get_random_inputs,
            # all exported methods must have the same signature so just pick the first one.
            methods[0],
        )

        # Create the ExportedModule.
        return ExportedModule(
            eager_module=eager_module,
            methods=methods,
            executorch_program=executorch_program,
            graph_module=graph_module,
            trace_inputs=trace_inputs,
            get_random_inputs_fn=get_random_inputs_fn,
        )
