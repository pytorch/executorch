# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Dict, List

import executorch.exir as exir
import torch
from executorch.exir import to_edge
from executorch.exir.backend.backend_api import LoweredBackendModule, to_backend
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)

# import the backend implementation
from executorch.exir.backend.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)
from executorch.exir.backend.test.hta_partitioner_demo import (
    HTAPartitionerMultiplePatternsDemo,
    HTAPartitionerOnePatternDemo,
)
from executorch.exir.backend.test.op_partitioner_demo import (
    AddAttributePartitionerDemo,
    AddMulPartitionerDemo,
)
from executorch.exir.backend.test.qnn_backend_demo import QnnBackend

from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.graph_module import get_control_flow_submodules
from executorch.exir.lowered_backend_module import (
    get_lowered_backend_modules,
    get_lowered_submodules,
)
from executorch.exir.print_program import print_program
from executorch.exir.schema import (
    BackendDelegate,
    BackendDelegateDataReference,
    DataLocation,
    DelegateCall,
    Program,
)

from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from executorch.extension.pytree import tree_flatten

from functorch.experimental import control_flow
from torch.ao.quantization import get_default_qconfig_mapping  # @manual
from torch.ao.quantization.backend_config.executorch import (
    get_executorch_backend_config,
)
from torch.ao.quantization.quantize_fx import (
    _convert_to_reference_decomposed_fx,
    prepare_fx,
)
from torch.export import export, ExportedProgram
from torch.testing import FileCheck


def vary_segments(test_method):
    """A decorator that calls the test method with `extract_delegate_segments` set to
    True and False.

    Decorated test methods must expect a boolean parameter named
    `extract_delegate_segments`, and they should pass that value to to_executorch() like:

        m.to_executorch(
            config=exir.ExecutorchBackendConfig(extract_delegate_segments=extract_delegate_segments)
        )

    This will cause the delegate data blobs to be extracted from the program and
    serialized as separate, freeable program segments. Backends should detect no
    difference at runtime.
    """

    def wrapper(self):
        for extract_delegate_segments in [False, True]:
            # subTest will create a different top-level test entry for each
            # value, whose full names have a suffix like
            # "(extract_delegate_segments=True)".
            with self.subTest(extract_delegate_segments=extract_delegate_segments):
                test_method(self, extract_delegate_segments=extract_delegate_segments)

    return wrapper


class TestBackends(unittest.TestCase):
    def check_delegate_input(
        self, delegate: LoweredBackendModule, input_len: int
    ) -> None:
        counter = 0
        for node in delegate.original_module.graph.nodes:
            if node.op == "placeholder":
                counter += 1
        self.assertEqual(counter, input_len)

    def check_backend_delegate(
        self,
        program: Program,
        delegate: BackendDelegate,
        expected_id: str,
        expected_processed: bytes,
    ) -> None:
        self.assertEqual(delegate.id, expected_id)
        processed: BackendDelegateDataReference = delegate.processed
        self.assertEqual(processed.location, DataLocation.INLINE)
        self.assertLess(processed.index, len(program.backend_delegate_data))
        self.assertEqual(
            program.backend_delegate_data[processed.index].data, expected_processed
        )

    def test_simple(self):
        class SinModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.sin(x)

        sin_module = SinModule()
        model_inputs = (torch.ones(1),)
        expected_res = sin_module(*model_inputs)
        edgeir_m = to_edge(export(sin_module, model_inputs))

        lowered_sin_module = to_backend(
            "BackendWithCompilerDemo", edgeir_m.exported_program(), []
        )
        new_res = lowered_sin_module(*model_inputs)

        self.assertTrue(torch.allclose(new_res, expected_res))

        # TODO(tkaruturi): emitting single LoweredBackendModule
        # program = to_edge(export(graph_module)).to_exectorch()._emitter_output.program

    @vary_segments
    def test_backend_with_compiler(self, extract_delegate_segments: bool):
        class SinModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # TODO(chenlai): add a test with a diffrent method name when
            # it's resolved in compiler side.
            def forward(self, x):
                return torch.sin(x)

        sin_module = SinModule()
        model_inputs = (torch.ones(1),)
        edgeir_m = to_edge(export(sin_module, model_inputs))
        max_value = model_inputs[0].shape[0]
        compile_specs = [CompileSpec("max_value", bytes([max_value]))]
        lowered_sin_module = to_backend(
            "BackendWithCompilerDemo", edgeir_m.exported_program(), compile_specs
        )

        class CompositeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered_linear_sin = lowered_sin_module

            def forward(self, x):
                return self.lowered_linear_sin(x)

        composite_model = CompositeModule()
        model_inputs = (torch.ones(1),)

        composite_model(*model_inputs)

        exec_prog = to_edge(export(composite_model, model_inputs)).to_executorch(
            config=exir.ExecutorchBackendConfig(
                extract_delegate_segments=extract_delegate_segments
            )
        )
        graph_module = exec_prog.exported_program().graph_module

        # Check that there is not an aten.sin node.
        self.assertTrue(
            exir_ops.edge.aten.sin
            not in {node.target for node in graph_module.graph.nodes}
        )

        # Check that there exists a call_delegate, representing the call to the
        # delegated function
        FileCheck().check("torch.ops.higher_order.executorch_call_delegate").run(
            graph_module.code
        )
        lowered_submodules = get_lowered_submodules(graph_module)
        self.assertEqual(len(lowered_submodules), 1)

        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target == executorch_call_delegate:
                # Check that first arg is lowered_module_{unique_id}
                self.assertEqual(node.args[0].target, "lowered_module_0")

        program = exec_prog._emitter_output.program

        # Check the program can be printed
        print_program(program)

        # Check the backend delegate
        self.check_backend_delegate(
            program=program,
            delegate=program.execution_plan[0].delegates[0],
            expected_id=BackendWithCompilerDemo.__name__,
            expected_processed=b"1version:0#op:demo::aten.sin.default, numel:1, dtype:torch.float32<debug_handle>1#",
        )

        # Check the delegate instruction
        self.assertTrue(
            isinstance(
                program.execution_plan[0].chains[0].instructions[0].instr_args,
                DelegateCall,
            )
        )
        buff = exec_prog.buffer

        executorch_module = _load_for_executorch_from_buffer(buff)
        model_inputs = torch.ones(1)
        model_outputs = executorch_module.forward([model_inputs])
        self.assertEqual(
            model_inputs,
            torch.ones(1),
        )
        expected_output = 0.8333 * torch.ones(1)

        self.assertTrue(
            torch.allclose(model_outputs[0], expected_output, atol=1e-03, rtol=1e-03)
        )

    @vary_segments
    def test_lowered_add_mul(self, extract_delegate_segments: bool):
        class AddMulModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = torch.add(y, b)
                return z

        add_mul_module = AddMulModule()
        model_inputs = (torch.ones(2, 2), 2 * torch.ones(2, 2), 3 * torch.ones(2, 2))
        edge_graph_module = to_edge(export(add_mul_module, model_inputs))
        max_value = model_inputs[0].shape[0]
        compile_specs = [CompileSpec("max_value", bytes([max_value]))]
        lowered_add_mul = to_backend(
            "BackendWithCompilerDemo",
            edge_graph_module.exported_program(),
            compile_specs,
        )

        class CompositeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered_add_mul = lowered_add_mul

            def forward(self, a, x, b):
                return self.lowered_add_mul(a, x, b)

        composite_model = CompositeModule()

        composite_model(*model_inputs)

        exec_prog = to_edge(export(composite_model, model_inputs)).to_executorch(
            config=exir.ExecutorchBackendConfig(
                extract_delegate_segments=extract_delegate_segments
            )
        )
        buff = exec_prog.buffer

        executorch_module = _load_for_executorch_from_buffer(buff)

        # pyre-fixme[16]: Module `pytree` has no attribute `tree_flatten`.
        inputs_flattened, _ = tree_flatten(model_inputs)
        model_output = executorch_module.run_method("forward", tuple(inputs_flattened))
        ref_output = add_mul_module(*model_inputs)

        self.assertTrue(
            torch.allclose(model_output[0], ref_output, atol=1e-03, rtol=1e-03)
        )

    def run_model_in_unsupported_backend(self, extract_delegate_segments: bool):
        class SinModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.sin(x)

        sin_module = SinModule()
        # the backend only  accepts shape <= 4
        model_inputs = (torch.ones(6),)
        edgeir_m = to_edge(export(sin_module, model_inputs))
        max_value = model_inputs[0].shape[0]
        compile_specs = [CompileSpec("max_value", bytes([max_value]))]
        lowered_sin_module = to_backend(
            "BackendWithCompilerDemo", edgeir_m.exported_program(), compile_specs
        )

        class CompositeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered_linear_sin = lowered_sin_module

            def forward(self, x):
                return self.lowered_linear_sin(x)

        composite_model = CompositeModule()
        model_inputs = (torch.zeros(6),)

        composite_model(*model_inputs)

        exec_prog = to_edge(export(composite_model, model_inputs)).to_executorch(
            config=exir.ExecutorchBackendConfig(
                extract_delegate_segments=extract_delegate_segments
            ),
        )

        buff = exec_prog.buffer

        # This line should raise an exception like
        # RuntimeError: failed with error 0x12
        _load_for_executorch_from_buffer(buff)

    @vary_segments
    def test_backend_with_compiler_out_of_range(self, extract_delegate_segments: bool):
        with self.assertRaisesRegex(
            RuntimeError,
            "loading method forward failed with error 0x12",
        ):
            self.run_model_in_unsupported_backend(
                extract_delegate_segments=extract_delegate_segments
            )

    @vary_segments
    def test_backend_with_compiler_delegate_and_operator(
        self, extract_delegate_segments: bool
    ):
        # Test includes both delegates and operator
        # import the backend implementation
        from executorch.exir.backend.test.backend_with_compiler_demo import (
            BackendWithCompilerDemo,
        )

        class SinModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # TODO(chenlai): add a test with a diffrent method name when
            # it's resolved in compiler side.
            def forward(self, x):
                return [torch.sin(x)]

        sin_module = SinModule()
        model_inputs = (torch.ones(1),)
        edgeir_m = to_edge(export(sin_module, model_inputs))
        max_value = model_inputs[0].shape[0]
        compile_specs = [CompileSpec("max_value", bytes([max_value]))]
        lowered_sin_module = to_backend(
            "BackendWithCompilerDemo", edgeir_m.exported_program(), compile_specs
        )

        class CompositeModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered_linear_sin = lowered_sin_module

            def forward(self, x):
                a = self.lowered_linear_sin(x)[0]
                b = self.lowered_linear_sin(x)[0]
                return torch.add(a, b)

        composite_model = CompositeModule()
        model_inputs = (torch.ones(1),)

        composite_model(*model_inputs)

        exec_prog = to_edge(export(composite_model, model_inputs)).to_executorch(
            config=exir.ExecutorchBackendConfig(
                extract_delegate_segments=extract_delegate_segments
            ),
        )
        graph_module = exec_prog.exported_program().graph_module
        program = exec_prog._emitter_output.program
        buff = exec_prog.buffer

        # Check that there is not an aten.sin node.
        self.assertTrue(
            exir_ops.edge.aten.sin.default
            not in {node.target for node in graph_module.graph.nodes}
        )

        # Check that there exists a call_delegate op, representing the call to the
        # delegated function
        FileCheck().check("torch.ops.higher_order.executorch_call_delegate").run(
            graph_module.code
        )

        for node in graph_module.graph.nodes:
            if node.op == "call_function" and node.target == executorch_call_delegate:
                # Check that first arg is lowered_module_{unique_id}
                self.assertEqual(node.args[0].target, "lowered_module_0")

        # Check the backend delegate
        self.check_backend_delegate(
            program=program,
            delegate=program.execution_plan[0].delegates[0],
            expected_id=BackendWithCompilerDemo.__name__,
            expected_processed=b"1version:0#op:demo::aten.sin.default, numel:1, dtype:torch.float32<debug_handle>1#",
        )

        # Check the delegate instruction
        self.assertTrue(
            isinstance(
                program.execution_plan[0].chains[0].instructions[0].instr_args,
                DelegateCall,
            )
        )

        executorch_module = _load_for_executorch_from_buffer(buff)
        model_inputs = torch.ones(1)

        model_outputs = executorch_module.forward([model_inputs])

        self.assertEqual(
            model_inputs,
            torch.ones(1),
        )
        expected_output = 1.666667 * torch.ones(1)

        self.assertTrue(
            torch.allclose(model_outputs[0], expected_output, atol=1e-03, rtol=1e-03)
        )

    def test_backend_with_compiler_backend_runtime_exception(self):
        class SinModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # TODO(chenlai): add a test with a diffrent method name when
            # it's resolved in compiler side.
            def forward(self, x):
                return torch.sin(x) + torch.cos(x)

        sin_module = SinModule()
        model_inputs = (torch.ones(1),)
        edgeir_m = to_edge(export(sin_module, model_inputs))
        error_msg = r"call_function aten.cos.default is not supported in backend BackendWithCompilerDemo"

        with self.assertRaisesRegex(
            RuntimeError,
            error_msg,
        ):
            _ = to_backend("BackendWithCompilerDemo", edgeir_m.exported_program(), [])

    def test_backend_with_compiler_backend_not_found_exception(self):
        class SinModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            # TODO(chenlai): add a test with a diffrent method name when
            # it's resolved in compiler side.
            def forward(self, x):
                return torch.sin(x) + torch.cos(x)

        sin_module = SinModule()
        model_inputs = (torch.ones(1),)
        edgeir_m = to_edge(export(sin_module, model_inputs))
        error_msg = r"Backend FakeBackendWithCompilerDemo was not found."

        with self.assertRaisesRegex(
            NotImplementedError,
            error_msg,
        ):
            _ = to_backend(
                "FakeBackendWithCompilerDemo", edgeir_m.exported_program(), []
            )

    @vary_segments
    def test_backend_with_compiler_delegate_and_operator_with_two_modules(
        self, extract_delegate_segments: bool
    ):
        # the submodule runs in a specific backend. In this example, `BackendWithCompilerDemo` backend
        class LowerableSubModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.sin(x)

        # sin_module is an nn.Module
        to_be_lowered = LowerableSubModel()
        example_input = (torch.ones(1),)
        to_be_lowered_exir_submodule = to_edge(export(to_be_lowered, example_input))

        max_value = example_input[0].shape[0]
        compile_specs = [CompileSpec("max_value", bytes([max_value]))]
        lowered_module = to_backend(
            "BackendWithCompilerDemo",
            to_be_lowered_exir_submodule.exported_program(),
            compile_specs,
        )

        class NonLowerableSubModel(torch.nn.Module):
            def __init__(self, bias):
                super().__init__()
                self.register_buffer("bias", bias)

            def forward(self, a, b):
                return torch.add(torch.add(a, b), self.bias)

        # the composite modules, including lower part and non-lowerpart
        class CompositeModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.non_lowerable = NonLowerableSubModel(torch.ones(1) * 0.3)
                self.lowerable = lowered_module

            def forward(self, x):
                a = self.lowerable(x)
                b = self.lowerable(a)
                ret = self.non_lowerable(a, b)
                return a, b, ret

        composite_model = CompositeModel()

        # Prepare the model input
        model_inputs = (torch.ones(1),)

        # Verify the input works with eager module
        composite_model(*model_inputs)

        exec_prog = to_edge(export(composite_model, model_inputs)).to_executorch(
            config=exir.ExecutorchBackendConfig(
                extract_delegate_segments=extract_delegate_segments
            ),
        )
        flatbuffer = exec_prog.buffer

        executorch_module = _load_for_executorch_from_buffer(flatbuffer)
        model_outputs = executorch_module.forward([*model_inputs])

        expected_outputs = [
            0.8333 * torch.ones(1),
            0.7369 * torch.ones(1),
            1.8702 * torch.ones(1),
        ]

        for index, expected_output in enumerate(expected_outputs):
            self.assertTrue(
                torch.allclose(
                    model_outputs[index], expected_output, atol=1e-03, rtol=1e-03
                )
            )

    @vary_segments
    def test_partition_delegate_graph_with_multiple_patterns(
        self, extract_delegate_segments: bool
    ):
        class CompositeModel(torch.nn.Module):
            def __init__(self, _weight):
                super().__init__()
                self.weight = _weight
                self.lstm = torch.nn.LSTM(
                    input_size=32,
                    hidden_size=32,
                    num_layers=1,
                )
                self.conv = torch.nn.Conv1d(1, 1, 1, stride=2)

            def forward(self, x_raw, h, c):
                output, (hn, cn) = self.lstm(x_raw, (h, c))
                k = self.conv(output)
                x = output
                y = cn
                a = torch.sub(x, y)
                b = torch.sub(x, a)
                c = torch.sub(x, b)
                d = torch.add(x, self.weight)
                e = torch.mul(c, d)
                return e, hn, k

        # Prepare input and trace it
        input_x = torch.ones([1, 32])
        input_h = torch.ones([1, 32])
        input_c = torch.ones([1, 32])
        inputs = (input_x, input_h, input_c)

        composite_m = CompositeModel(3)
        orig_res = composite_m(*inputs)

        traced = to_edge(
            export(composite_m, inputs),
            compile_config=exir.EdgeCompileConfig(
                _check_ir_validity=False, _use_edge_ops=True
            ),
        )

        program_without_delegates = to_edge(
            export(CompositeModel(3), inputs),
            compile_config=exir.EdgeCompileConfig(
                _check_ir_validity=False,
            ),
        ).to_executorch(
            config=exir.ExecutorchBackendConfig(
                extract_delegate_segments=extract_delegate_segments
            ),
        )
        # after this step, part of the graph will be lowered to backend, depending on
        # HTAPartitionerDemo's rule.
        program_with_delegates = traced
        program_with_delegates = program_with_delegates.to_backend(
            HTAPartitionerMultiplePatternsDemo()
        )
        program_with_delegates = program_with_delegates.to_executorch(
            config=exir.ExecutorchBackendConfig(
                extract_delegate_segments=extract_delegate_segments
            ),
        )

        new_res = program_with_delegates.exported_program().module()(*inputs)
        for t1, t2 in zip(new_res, orig_res, strict=True):
            self.assertTrue(torch.allclose(t1, t2, atol=1e-03, rtol=1e-03))

        # Check the backend delegate
        self.check_backend_delegate(
            program=program_with_delegates._emitter_output.program,
            delegate=program_with_delegates._emitter_output.program.execution_plan[
                0
            ].delegates[0],
            expected_id=QnnBackend.__name__,
            expected_processed=b"imqnncompiled",
        )

        # Check add not in the program with delegates
        self.assertEqual(
            0,
            len(
                [
                    op
                    for op in program_with_delegates._emitter_output.program.execution_plan[
                        0
                    ].operators
                    if op.name == "aten::sub"
                ]
            ),
        )

        # Check convolution not in the program with delegates
        self.assertEqual(
            0,
            len(
                [
                    op
                    for op in program_with_delegates._emitter_output.program.execution_plan[
                        0
                    ].operators
                    if op.name == "aten::convolution"
                ]
            ),
        )

        # Check convolution in the program without delegates
        self.assertEqual(
            1,
            len(
                [
                    op
                    for op in program_without_delegates._emitter_output.program.execution_plan[
                        0
                    ].operators
                    if op.name == "aten::convolution"
                ]
            ),
        )

    @vary_segments
    def test_partition_delegate_graph_with_one_patterns(
        self, extract_delegate_segments: bool
    ):
        class CompositeModel(torch.nn.Module):
            def __init__(self, _weight):
                super().__init__()
                self.weight = _weight
                self.lstm = torch.nn.LSTM(
                    input_size=32,
                    hidden_size=32,
                    num_layers=1,
                )
                self.conv = torch.nn.Conv1d(1, 1, 1, stride=2)

            def forward(self, x_raw, h, c):
                output, (hn, cn) = self.lstm(x_raw, (h, c))
                k = self.conv(output)
                x = output
                y = cn
                a = torch.sub(x, y)
                b = torch.sub(x, a)
                c = torch.sub(x, b)
                d = torch.add(x, self.weight)
                e = torch.mul(c, d)
                return e, hn, k

        # Prepare input and trace it
        input_x = torch.ones([1, 32])
        input_h = torch.ones([1, 32])
        input_c = torch.ones([1, 32])
        inputs = (input_x, input_h, input_c)

        composite_m = CompositeModel(3)
        orig_res = composite_m(*inputs)

        traced = to_edge(
            export(composite_m, inputs),
            compile_config=exir.EdgeCompileConfig(
                _check_ir_validity=False, _use_edge_ops=True
            ),
        )

        program_without_delegates = to_edge(
            export(
                CompositeModel(3),
                (input_x, input_h, input_c),
            ),
            compile_config=exir.EdgeCompileConfig(
                _check_ir_validity=False,
            ),
        ).to_executorch(
            config=exir.ExecutorchBackendConfig(
                extract_delegate_segments=extract_delegate_segments
            ),
        )
        # after this step, part of the graph will be lowered to backend, depending on
        # HTAPartitionerDemo's rule.
        traced_with_delegate = traced
        traced_with_delegate = traced_with_delegate.to_backend(
            HTAPartitionerOnePatternDemo()
        )

        new_res = traced_with_delegate.exported_program().module()(*inputs)
        for t1, t2 in zip(new_res, orig_res, strict=True):
            self.assertTrue(torch.allclose(t1, t2, atol=1e-03, rtol=1e-03))

        program_with_delegates = traced_with_delegate.to_executorch(
            config=exir.ExecutorchBackendConfig(
                extract_delegate_segments=extract_delegate_segments
            ),
        )

        # TODO(T143084047): Currently not retraceable
        # Retracing is not needed, but keeping this here to make sure the result
        # of to_backend is retraceable
        # graph_module_with_delegate = to_edge(export(
        #     traced_with_delegate,
        #     (input_x, input_h, input_c),
        #
        # ))

        # program_with_delegates = graph_module_with_delegate.to_executorch(
        #     config=exir.ExecutorchBackendConfig(extract_delegate_segments=extract_delegate_segments),
        # )

        new_res = program_with_delegates.exported_program().module()(*inputs)
        for t1, t2 in zip(new_res, orig_res, strict=True):
            self.assertTrue(torch.allclose(t1, t2, atol=1e-03, rtol=1e-03))

        # Check the backend delegate
        self.check_backend_delegate(
            program=program_with_delegates._emitter_output.program,
            delegate=program_with_delegates._emitter_output.program.execution_plan[
                0
            ].delegates[0],
            expected_id=QnnBackend.__name__,
            expected_processed=b"imqnncompiled",
        )

        # Check add is in the program with delegates
        self.assertEqual(
            1,
            len(
                [
                    op
                    for op in program_with_delegates._emitter_output.program.execution_plan[
                        0
                    ].operators
                    if op.name == "aten::sub"
                ]
            ),
        )

        # Check convolution not in the program with delegates
        self.assertEqual(
            0,
            len(
                [
                    op
                    for op in program_with_delegates._emitter_output.program.execution_plan[
                        0
                    ].operators
                    if op.name == "aten::convolution"
                ]
            ),
        )

        # Check convolution in the program without delegates
        self.assertEqual(
            1,
            len(
                [
                    op
                    for op in program_without_delegates._emitter_output.program.execution_plan[
                        0
                    ].operators
                    if op.name == "aten::convolution"
                ]
            ),
        )

    @vary_segments
    def test_add_mul_partitioner(self, extract_delegate_segments: bool):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, a, x, b):
                y = torch.mm(a, x)
                z = y + b
                a = z - a
                y = torch.mm(a, x)
                z = y + b
                return z

        m = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))
        orig_res = m(*inputs)

        ep = to_edge(export(m, inputs))
        executorch_prog = ep
        executorch_prog = executorch_prog.to_backend(AddMulPartitionerDemo())
        executorch_prog = executorch_prog.to_executorch(
            config=exir.ExecutorchBackendConfig(
                extract_delegate_segments=extract_delegate_segments
            ),
        )

        new_res = executorch_prog.exported_program().graph_module(*inputs)
        self.assertTrue(torch.allclose(new_res[0], orig_res))

        counter = 0
        for node in executorch_prog.exported_program().graph_module.graph.nodes:
            if node.op == "get_attr":
                self.assertEqual(node.target, f"lowered_module_{counter}")
                counter += 1
        # There should be 2 delegated modules
        self.assertEqual(counter, 2)

        executorch_module = _load_for_executorch_from_buffer(executorch_prog.buffer)
        # pyre-fixme[16]: Module `pytree` has no attribute `tree_flatten`.
        inputs_flattened, _ = tree_flatten(inputs)
        model_output = executorch_module.run_method("forward", tuple(inputs_flattened))
        ref_output = m(*inputs)

        self.assertTrue(
            torch.allclose(model_output[0], ref_output, atol=1e-03, rtol=1e-03),
        )

    @vary_segments
    def test_partitioner_with_attributes(self, extract_delegate_segments: bool):
        """
        check that parameters that are lowered are correctly moved into the sub
        program, rather than being retained and passed as inputs.
        """

        class AddOne(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("one", torch.ones(1, 3))

            def forward(self, x):
                return x + self.one

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.add_one = AddOne()
                self.add_one_2 = AddOne()

            def forward(self, x, y):
                x = self.add_one(x) * y
                return self.add_one_2(x)

        inputs = (torch.randn(1, 3), torch.randn(1, 3))
        orig_res = Model()(*inputs)
        ep = to_edge(export(Model(), inputs))
        executorch_prog = ep
        executorch_prog = executorch_prog.to_backend(AddAttributePartitionerDemo())
        executorch_prog = executorch_prog.to_executorch(
            config=exir.ExecutorchBackendConfig(
                extract_delegate_segments=extract_delegate_segments
            ),
        )

        # Check the delegated submodules
        lowered_backends = get_lowered_backend_modules(
            executorch_prog.exported_program().graph_module
        )
        self.assertEqual(len(lowered_backends), 2)
        for backend in lowered_backends:
            original_program = backend.original_module
            # check that program has the lowered attributes
            self.assertEqual(len(original_program.state_dict), 1)
            # check backend has one placeholder input one placeholder parameter
            self.check_delegate_input(backend, 2)

        executorch_prog.buffer

        new_res = executorch_prog.exported_program().graph_module(*inputs)
        self.assertTrue(torch.allclose(orig_res, new_res[0]))

    def test_bad_partitioner(self):
        """
        Checks that we throw an error if user provided partitioner modifies the
        graph module
        """
        inputs = (torch.randn(1, 3), torch.randn(1, 3))

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                x = x + y
                x = x * y
                x = x - y
                x = x / y
                x = x * y
                x = x + y
                return x

        class BadPartitioner(Partitioner):
            partition_tags = {"tag1": DelegationSpec("BackendWithCompilerDemo", [])}

            def partition(self, exported_program: ExportedProgram) -> PartitionResult:
                # Partitioner should not modify the given graph module
                partition_tags: Dict[str, DelegationSpec] = {}
                for node in exported_program.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == exir_ops.edge.aten.add.Tensor
                    ):
                        node.target = exir_ops.edge.aten.mul.Tensor
                return PartitionResult(
                    tagged_exported_program=exported_program,
                    partition_tags=partition_tags,
                )

        ep = to_edge(export(Model(), inputs))
        with self.assertRaises(AssertionError):
            _ = ep.to_backend(BadPartitioner())

    def test_quantized_with_delegate(self) -> None:
        torch.ops.load_library(
            "//executorch/kernels/quantized:custom_ops_generated_lib"
        )
        qconfig_mapping = get_default_qconfig_mapping("qnnpack")
        in_size = 2
        input_size = 3
        output_size = 4
        linear = torch.nn.Linear(input_size, output_size).eval()
        example_inputs = (torch.ones(in_size, input_size),)
        prepared_linear = prepare_fx(
            linear,
            qconfig_mapping,
            example_inputs,
            backend_config=get_executorch_backend_config(),
        )
        converted_linear: torch.nn.Module = _convert_to_reference_decomposed_fx(
            prepared_linear,
        )

        # fails to trace here
        converted_linear_gm = to_edge(
            export(
                converted_linear,
                example_inputs,
            ),
            compile_config=exir.EdgeCompileConfig(
                _check_ir_validity=False,
            ),
        )
        FileCheck().check_count("quantize_per_tensor_default", 3).check("addmm").run(
            converted_linear_gm.exported_program().graph_module.code
        )

    def test_partition_with_control_flow(self) -> None:
        def true_fn(x, y):
            x = x - y
            x = x + y
            x = x - y
            return x

        def false_fn(x, y):
            x = x - y
            x = torch.mm(x, y)
            x = x - y
            return x

        class Module(torch.nn.Module):
            def forward(self, x, y):
                x = x + y
                x = control_flow.cond(x[0][0] == 1, true_fn, false_fn, [x, y])
                x = x - y
                return x

        f = Module()
        inputs = (torch.ones(2, 2), torch.ones(2, 2))
        orig_res = f(*inputs)
        orig = to_edge(
            export(
                f,
                inputs,
            )
        )
        partitioned = orig
        partitioned = partitioned.to_backend(AddMulPartitionerDemo())

        new_res = partitioned.exported_program().module()(*inputs)
        self.assertTrue(torch.allclose(orig_res, new_res[0]))

        toplevel_lowered = get_lowered_submodules(
            partitioned.exported_program().graph_module
        )
        self.assertEqual(len(toplevel_lowered), 1)
        FileCheck().check("executorch_exir_dialects_edge__ops_aten_add_Tensor").run(
            toplevel_lowered[0][1].original_module.graph_module.code
        )

        # Toplevel module only has the cond submodules
        partitioned_submodules = get_control_flow_submodules(
            partitioned.exported_program().graph_module
        )
        self.assertEqual(len(partitioned_submodules), 2)

        true_gm = partitioned_submodules[0][1]
        true_lowered = get_lowered_submodules(true_gm)
        self.assertEqual(len(true_lowered), 1)
        FileCheck().check("executorch_exir_dialects_edge__ops_aten_add_Tensor").run(
            true_lowered[0][1].original_module.graph_module.code
        )

        false_gm = partitioned_submodules[1][1]
        false_lowered = get_lowered_submodules(false_gm)
        self.assertEqual(len(true_lowered), 1)
        FileCheck().check("executorch_exir_dialects_edge__ops_aten_mm_default").run(
            false_lowered[0][1].original_module.graph_module.code
        )

    def test_partition_with_map(self) -> None:
        def map_fn(x, y):
            x = x - y
            x = x + y
            return x

        class Module(torch.nn.Module):
            def forward(self, xs, y):
                y = torch.mm(y, y)
                return control_flow.map(map_fn, xs, y)

        f = Module()
        inputs = (torch.ones(2, 2), torch.ones(2, 2))
        orig_res = f(*inputs)
        orig = to_edge(
            export(
                f,
                inputs,
            )
        )
        partitioned = orig
        partitioned = partitioned.to_backend(AddMulPartitionerDemo())

        toplevel_lowered = get_lowered_submodules(
            partitioned.exported_program().graph_module
        )
        self.assertEqual(len(toplevel_lowered), 1)
        FileCheck().check("executorch_exir_dialects_edge__ops_aten_mm_default").run(
            toplevel_lowered[0][1].original_module.graph_module.code
        )

        # Toplevel module only has the map submodule
        partitioned_submodules = get_control_flow_submodules(
            partitioned.exported_program().graph_module
        )
        self.assertEqual(len(partitioned_submodules), 1)

        map_fn_gm = partitioned_submodules[0][1]
        map_fn_lowered = get_lowered_submodules(map_fn_gm)
        self.assertEqual(len(map_fn_lowered), 1)
        FileCheck().check("executorch_exir_dialects_edge__ops_aten_add_Tensor").run(
            map_fn_lowered[0][1].original_module.graph_module.code
        )

        new_res = partitioned.exported_program().module()(*inputs)

        self.assertTrue(torch.allclose(orig_res, new_res[0]))

    def test_partition_with_nested_control_flow(self) -> None:
        """
        Partitions the add and mul ops, including the ones inside the submodules
        """

        def true_nested(y):
            y = y + y
            y = torch.mm(y, y)
            return y

        def false_nested(y):
            return torch.mm(y, y)

        def true_fn(x, pred2):
            z = control_flow.cond(pred2, true_nested, false_nested, [x])
            return x + z

        def false_fn(x, _):
            return x.cos()

        def map_fn(x, pred1, pred2, y):
            x = x.cos()
            y = control_flow.cond(pred1, true_fn, false_fn, [y, pred2])
            x = x + y
            return x.sin()

        class Module(torch.nn.Module):
            def forward(self, xs, pred1, pred2, y):
                y = torch.mm(y, y)
                return control_flow.map(map_fn, xs, pred1, pred2, y)

        inputs = (
            torch.ones(2, 2),
            torch.tensor([False]),
            torch.Tensor([False]),
            torch.ones(2, 2),
        )

        f = Module()
        orig_res = f(*inputs)
        orig = to_edge(
            export(
                f,
                inputs,
            )
        )
        partitioned = orig
        partitioned = partitioned.to_backend(AddMulPartitionerDemo())

        new_res = partitioned.exported_program().module()(*inputs)
        self.assertTrue(torch.allclose(orig_res, new_res[0]))

        toplevel_lowered = get_lowered_submodules(
            partitioned.exported_program().graph_module
        )
        self.assertEqual(len(toplevel_lowered), 1)
        FileCheck().check("executorch_exir_dialects_edge__ops_aten_mm_default").run(
            toplevel_lowered[0][1].original_module.graph_module.code
        )

        # Toplevel module only has the map submodule
        partitioned_submodules = get_control_flow_submodules(
            partitioned.exported_program().graph_module
        )
        self.assertEqual(len(partitioned_submodules), 1)

        # Map module has the cond submodules
        map_submodules = get_control_flow_submodules(partitioned_submodules[0][1])
        self.assertEqual(len(map_submodules), 2)

        # True module
        true_module = map_submodules[0][1]
        true_lowered = get_lowered_submodules(true_module)
        self.assertEqual(len(true_lowered), 1)
        FileCheck().check("executorch_exir_dialects_edge__ops_aten_add_Tensor").run(
            true_lowered[0][1].original_module.graph_module.code
        )

        # False module
        false_lowered = get_lowered_submodules(map_submodules[1][1])
        self.assertEqual(len(false_lowered), 0)

        # True module has the nested cond submodules
        true_submodules = get_control_flow_submodules(true_module)
        self.assertEqual(len(true_submodules), 2)

        # Nested True module
        true_true_lowered = get_lowered_submodules(true_submodules[0][1])
        self.assertEqual(len(true_true_lowered), 1)
        FileCheck().check("executorch_exir_dialects_edge__ops_aten_add_Tensor").check(
            "executorch_exir_dialects_edge__ops_aten_mm_default"
        ).run(true_true_lowered[0][1].original_module.graph_module.code)

        # Nested False module
        true_false_lowered = get_lowered_submodules(true_submodules[1][1])
        self.assertEqual(len(true_false_lowered), 1)
        FileCheck().check("executorch_exir_dialects_edge__ops_aten_mm_default").run(
            true_false_lowered[0][1].original_module.graph_module.code
        )

    def test_list_input(self):
        class Module(torch.nn.Module):
            def forward(self, x: List[torch.Tensor]):
                y = x[0] + x[1]
                return y

        f = Module()
        inputs = ([torch.randn(2, 2), torch.randn(2, 2)],)
        edge_prog = to_edge(export(f, inputs))
        lowered_gm = to_backend(
            BackendWithCompilerDemo.__name__, edge_prog.exported_program(), []
        )

        class ComposedM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered = lowered_gm

            def forward(self, x: List[torch.Tensor]):
                return self.lowered(x)

        gm = to_edge(export(ComposedM(), inputs))
        gm.exported_program().module()(*inputs)

    def test_dict_input(self):
        class Module(torch.nn.Module):
            def forward(self, x: Dict[str, torch.Tensor]):
                y = x["a"] + x["b"]
                return y

        f = Module()
        inputs = ({"a": torch.randn(2, 2), "b": torch.randn(2, 2)},)
        edge_prog = to_edge(export(f, inputs))
        lowered_gm = to_backend(
            BackendWithCompilerDemo.__name__, edge_prog.exported_program(), []
        )

        class ComposedM(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered = lowered_gm

            def forward(self, x: List[torch.Tensor]):
                return self.lowered(x)

        gm = to_edge(export(ComposedM(), inputs))
        gm.exported_program().module()(*inputs)
