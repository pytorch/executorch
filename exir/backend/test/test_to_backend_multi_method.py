# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Dict, List, Tuple

import torch

from executorch.exir import EdgeProgramManager, to_edge
from executorch.exir.backend.backend_api import (
    MethodProgramsPartitionerSpec,
    to_backend,
)

from executorch.exir.backend.canonical_partitioners.all_node_partitioner import (
    AllNodePartitioner,
)
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import Partitioner
from executorch.exir.backend.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)

from executorch.exir.backend.test.backend_with_preprocess_all_demo import (
    BackendWithPreprocessAllPartitioner,
)
from executorch.exir.graph_module import get_control_flow_submodules
from executorch.exir.lowered_backend_module import (
    get_lowered_submodules,
    LoweredBackendModule,
)
from executorch.exir.schema import (
    BackendDelegate,
    BackendDelegateDataReference,
    DataLocation,
    Program,
)
from executorch.extension.pybindings.portable_lib import (  # @manual
    _load_for_executorch_from_buffer,
)
from torch.export.exported_program import ExportedProgram

from torch.testing import FileCheck


class TestToBackendMultiMethod(unittest.TestCase):
    """
    Testing suite used to test multi method to_backend lowering. The test suite uses demo backends
    FirstBackendWithPreprocessAll and SecondBackendWithPreprocessAll.
    - FirstBackendWithPreprocessAll: supports add + sin
    - SecondBackendWithPreprocessAll: supports sub + cos

    Both backends lower exported programs into payloads in the string format:
    - (backend_id)#(total_number_ops across methods)#[op_target_name;]#[compile_spec.key:compile_spec.value;]

    We leverage the above expectation to test various lowering across different modules, and ensure
    that the right exported programs and compile specs are given when lowering a specifed exported program

    We leverage the demo partitioner BackendWithPreprocessAll which partitions add + sin nodes to
    FirstBackendWithPreprocessAll and sub + cos nodes to SecondBackendWithPreprocessAll. This allows
    us to test cases in which multiple backends are being lowered.
    """

    def _get_lowered_submodules_across_controlflow(
        self, graph_module: torch.fx.GraphModule
    ) -> List[Tuple[str, LoweredBackendModule, torch.fx.Node]]:
        top_level_submodules = get_lowered_submodules(graph_module)

        for _, submodule, _ in get_control_flow_submodules(graph_module):
            top_level_submodules.extend(
                self._get_lowered_submodules_across_controlflow(submodule)
            )

        return top_level_submodules

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

    def _test(
        self, test_set: Dict[str, Tuple[ExportedProgram, Partitioner, List[str]]]
    ):
        method_to_edge_program = {
            method_name: ep for method_name, (ep, _, _) in test_set.items()
        }

        method_to_partitioner = {
            method_name: partitioner
            for method_name, (_, partitioner, _) in test_set.items()
        }

        lowered_ep_dict = to_backend(
            MethodProgramsPartitionerSpec(
                method_to_edge_program,
                method_to_partitioner,
            )
        )

        self.assertEqual(len(lowered_ep_dict.keys()), len(test_set.keys()))
        for method_name in test_set.keys():
            self.assertTrue(method_name in lowered_ep_dict.keys())
            (_, _, list_of_payload_as_string) = test_set[method_name]
            lowered_ep = lowered_ep_dict[method_name]
            FileCheck().check_count(
                "torch.ops.higher_order.executorch_call_delegate",
                len(list_of_payload_as_string),
                exactly=True,
            ).run(str(lowered_ep))
            lowered_submodules = self._get_lowered_submodules_across_controlflow(
                lowered_ep.graph_module
            )
            self.assertEqual(len(lowered_submodules), len(list_of_payload_as_string))

            for idx, (_, lowered_backend_module, _) in enumerate(lowered_submodules):
                self.assertEqual(
                    lowered_backend_module.processed_bytes.decode("utf-8"),
                    list_of_payload_as_string[idx],
                )

    def test_multi_method_to_backend_single_method(self):
        class SinModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.sin(x)

        edgeir_m = to_edge(torch.export.export(SinModule(), (torch.ones(1),)))
        # Payload String:
        # [Number of Ops lowered across all methods/partitions]#OpTargetNames#CompileSpecs;
        test_set = {
            "forward": (
                edgeir_m.exported_program(),
                AllNodePartitioner(
                    "FirstBackendWithPreprocessAll",
                    [CompileSpec("max_value", bytes([1]))],
                ),
                [
                    "FirstBackendWithPreprocessAll#1#aten.sin.default:#max_value:b'\\x01';"
                ],
            )
        }
        self._test(test_set)

    def test_multi_method_to_backend_two_methods(self):
        class SinModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.sin(x)

        class AddModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + x

        sin_edgeir_m = to_edge(torch.export.export(SinModule(), (torch.ones(1),)))
        add_edgeir_m = to_edge(torch.export.export(AddModule(), (torch.ones(1),)))
        sin_partitioner = AllNodePartitioner(
            "FirstBackendWithPreprocessAll", [CompileSpec("sin", bytes([2]))]
        )
        add_partitioner = AllNodePartitioner(
            "FirstBackendWithPreprocessAll", [CompileSpec("add", bytes([3]))]
        )
        # Payload String:
        # [Number of Ops lowered across all methods/partitions]#OpTargetNames#CompileSpecs;
        test_set = {
            "sin": (
                sin_edgeir_m.exported_program(),
                sin_partitioner,
                ["FirstBackendWithPreprocessAll#2#aten.sin.default:#sin:b'\\x02';"],
            ),
            "add": (
                add_edgeir_m.exported_program(),
                add_partitioner,
                ["FirstBackendWithPreprocessAll#2#aten.add.Tensor:#add:b'\\x03';"],
            ),
        }
        self._test(test_set)

    def test_multi_method_to_backend_two_methods_multiple_partitions(self):
        class AddModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = x + x
                y = y * y
                y = y + y
                return y

        class SinModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.sin(x)
                y = y * y
                return torch.sin(y)

        add_edgeir_m = to_edge(torch.export.export(AddModule(), (torch.ones(1),)))
        sin_edgeir_m = to_edge(torch.export.export(SinModule(), (torch.ones(1),)))
        test_set = {
            "add": (
                add_edgeir_m.exported_program(),
                BackendWithPreprocessAllPartitioner(),
                [
                    "FirstBackendWithPreprocessAll#4#aten.add.Tensor:#add:b'\\x00';",
                    "FirstBackendWithPreprocessAll#4#aten.add.Tensor:#add:b'\\x00';",
                ],
            ),
            "sin": (
                sin_edgeir_m.exported_program(),
                BackendWithPreprocessAllPartitioner(),
                [
                    "FirstBackendWithPreprocessAll#4#aten.sin.default:#sin:b'\\x01';",
                    "FirstBackendWithPreprocessAll#4#aten.sin.default:#sin:b'\\x01';",
                ],
            ),
        }
        self._test(test_set)

    def test_multi_method_to_backend_two_methods_different_partitions(self):
        class AddSinModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = x + x
                y = y * y
                y = torch.sin(y)
                return y

        class SinAddModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.sin(x)
                y = y * y
                return y + y

        add_sin_edgeir_m = to_edge(
            torch.export.export(AddSinModule(), (torch.ones(1),))
        )
        sin_add_edgeir_m = to_edge(
            torch.export.export(SinAddModule(), (torch.ones(1),))
        )
        test_set = {
            "add_sin": (
                add_sin_edgeir_m.exported_program(),
                BackendWithPreprocessAllPartitioner(),
                [
                    "FirstBackendWithPreprocessAll#4#aten.add.Tensor:#add:b'\\x00';",
                    "FirstBackendWithPreprocessAll#4#aten.sin.default:#sin:b'\\x01';",
                ],
            ),
            "sin_add": (
                sin_add_edgeir_m.exported_program(),
                BackendWithPreprocessAllPartitioner(),
                [
                    "FirstBackendWithPreprocessAll#4#aten.sin.default:#sin:b'\\x01';",
                    "FirstBackendWithPreprocessAll#4#aten.add.Tensor:#add:b'\\x00';",
                ],
            ),
        }
        self._test(test_set)

    def test_multi_method_to_backend_two_methods_different_backends(self):
        class AddSinCosSubModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = x + x
                y = torch.sin(y)
                y = torch.cos(y)
                y = y - x
                return y

        class CosSubAddSinModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                y = torch.cos(x)
                y = y - x
                y = y + y
                y = torch.sin(y)
                return y

        first_second_edgeir_m = to_edge(
            torch.export.export(AddSinCosSubModule(), (torch.ones(1),))
        )
        second_first_edgeir_m = to_edge(
            torch.export.export(CosSubAddSinModule(), (torch.ones(1),))
        )
        test_set = {
            "first_second": (
                first_second_edgeir_m.exported_program(),
                BackendWithPreprocessAllPartitioner(),
                [
                    "FirstBackendWithPreprocessAll#4#aten.add.Tensor:aten.sin.default:#add:b'\\x00';sin:b'\\x01';",
                    "SecondBackendWithPreprocessAll#4#aten.cos.default:aten.sub.Tensor:#cos:b'\\x03';sub:b'\\x02';",
                ],
            ),
            "second_first": (
                second_first_edgeir_m.exported_program(),
                BackendWithPreprocessAllPartitioner(),
                [
                    "SecondBackendWithPreprocessAll#4#aten.cos.default:aten.sub.Tensor:#cos:b'\\x03';sub:b'\\x02';",
                    "FirstBackendWithPreprocessAll#4#aten.add.Tensor:aten.sin.default:#add:b'\\x00';sin:b'\\x01';",
                ],
            ),
        }
        self._test(test_set)

    def test_multi_method_to_backend_control_flow(self):
        class SinCosModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def true_fn(self, x):
                return torch.sin(x)

            def false_fn(self, x):
                return torch.cos(x)

            def forward(self, x):
                x = x + x
                return torch.cond(x > 0, self.true_fn, self.false_fn, [x])

        class SinAddModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def true_fn(self, x):
                return torch.sin(x)

            def false_fn(self, x):
                return x + x

            def forward(self, x):
                return torch.cond(x > 0, self.true_fn, self.false_fn, [x])

        sin_cos_edgeir_m = to_edge(
            torch.export.export(SinCosModule(), (torch.ones(1),))
        )
        sin_add_edgeir_m = to_edge(
            torch.export.export(SinAddModule(), (torch.ones(1),))
        )

        test_set = {
            "sin_cos": (
                sin_cos_edgeir_m.exported_program(),
                BackendWithPreprocessAllPartitioner(),
                [
                    "FirstBackendWithPreprocessAll#4#aten.add.Tensor:#add:b'\\x00';",
                    # True Module Partition
                    "FirstBackendWithPreprocessAll#4#aten.sin.default:#sin:b'\\x01';",
                    # False Module Partition
                    "SecondBackendWithPreprocessAll#1#aten.cos.default:#cos:b'\\x03';",
                ],
            ),
            "sin_add": (
                sin_add_edgeir_m.exported_program(),
                BackendWithPreprocessAllPartitioner(),
                [
                    # True Module Partition
                    "FirstBackendWithPreprocessAll#4#aten.sin.default:#sin:b'\\x01';",
                    # False Module Partition
                    "FirstBackendWithPreprocessAll#4#aten.add.Tensor:#add:b'\\x00';",
                ],
            ),
        }
        self._test(test_set)

    def test_multi_method_to_backend_sequential_delegates(self):
        class SequentialBackendModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z):
                # delegate one
                x = x - x
                y = y - y
                z = z - z
                # graph break
                a = x * y * z
                # delegate two uses outputs from delegate one and the
                # output from the graph break
                b = x + a
                b = b + z + a
                b = b + y + a
                return b

        module = SequentialBackendModule()
        example_inputs = (torch.ones(1), torch.ones(1), torch.ones(1))
        seq_edgeir_m = to_edge(torch.export.export(module, example_inputs))

        test_set = {
            "seq_edgeir": (
                seq_edgeir_m.exported_program(),
                BackendWithPreprocessAllPartitioner(),
                [
                    "SecondBackendWithPreprocessAll#3#aten.sub.Tensor:aten.sub.Tensor:aten.sub.Tensor:#sub:b'\\x02';sub:b'\\x02';sub:b'\\x02';",
                    "FirstBackendWithPreprocessAll#5#aten.add.Tensor:aten.add.Tensor:aten.add.Tensor:aten.add.Tensor:aten.add.Tensor:#add:b'\\x00';add:b'\\x00';add:b'\\x00';add:b'\\x00';add:b'\\x00';",
                ],
            ),
        }
        self._test(test_set)

    def test_multi_method_to_backend_constants(self):
        class SequentialBackendModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.const = torch.zeros(1)

            def forward(self, x, y, z):
                # delegate one
                x = x - x
                y = y - y
                z = z - z
                # graph break
                a = x * y * z * self.const
                # delegate two uses outputs from delegate one and the
                # output from the graph break
                b = x + self.const + a
                b = z + a + b
                b = y + a + b
                return b

        module = SequentialBackendModule()
        example_inputs = (torch.ones(1), torch.ones(1), torch.ones(1))
        seq_const_m = to_edge(torch.export.export(module, example_inputs))

        test_set = {
            "seq_const": (
                seq_const_m.exported_program(),
                BackendWithPreprocessAllPartitioner(),
                [
                    "SecondBackendWithPreprocessAll#3#aten.sub.Tensor:aten.sub.Tensor:aten.sub.Tensor:#sub:b'\\x02';sub:b'\\x02';sub:b'\\x02';",
                    "FirstBackendWithPreprocessAll#6#CONSTc_const_copy_0:aten.add.Tensor:aten.add.Tensor:aten.add.Tensor:aten.add.Tensor:aten.add.Tensor:aten.add.Tensor:#add:b'\\x00';add:b'\\x00';add:b'\\x00';add:b'\\x00';add:b'\\x00';add:b'\\x00';",
                ],
            ),
        }
        self._test(test_set)

    def test_multi_method_to_backend_not_found(self):
        class SinModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.sin(x)

        class AddModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x + x

        sin_edgeir_m = to_edge(torch.export.export(SinModule(), (torch.ones(1),)))
        add_edgeir_m = to_edge(torch.export.export(AddModule(), (torch.ones(1),)))
        sin_partitioner = AllNodePartitioner(
            "Invalid", [CompileSpec("sin", bytes([2]))]
        )
        add_partitioner = AllNodePartitioner(
            "FirstBackendWithPreprocessAll", [CompileSpec("add", bytes([3]))]
        )

        test_set = {
            "sin": (
                sin_edgeir_m.exported_program(),
                sin_partitioner,
                [],
            ),
            "add": (
                add_edgeir_m.exported_program(),
                add_partitioner,
                [],
            ),
        }
        with self.assertRaisesRegex(
            NotImplementedError, "Backend Invalid was not found."
        ):
            self._test(test_set)

    def test_multi_method_end_to_end(self):
        """
        Tests multi method lowering end-to-end. Lowers the same Sin Module for two methods
        "forward" and "forward_copy". Ensures that the lowered program has two delegates
        but only one serialized blob. Ensures that the lowered program runs correctly.
        """

        class SinModule(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return torch.sin(x)

        sin_edgeir_m = to_edge(torch.export.export(SinModule(), (torch.ones(1),)))
        sin_edgeir_m_copy = to_edge(torch.export.export(SinModule(), (torch.ones(1),)))

        method_edge_program = {
            "forward": sin_edgeir_m.exported_program(),
            "forward_copy": sin_edgeir_m_copy.exported_program(),
        }
        compile_specs = [CompileSpec("max_value", bytes([1]))]

        method_partitioner = {
            "forward": AllNodePartitioner(
                BackendWithCompilerDemo.__name__, compile_specs
            ),
            "forward_copy": AllNodePartitioner(
                BackendWithCompilerDemo.__name__, compile_specs
            ),
        }

        lowered_ep_dict = to_backend(
            MethodProgramsPartitionerSpec(
                method_edge_program,
                method_partitioner,
            )
        )

        new_edge_manager = EdgeProgramManager(lowered_ep_dict)

        exec_prog = new_edge_manager.to_executorch()

        program = exec_prog.executorch_program
        # Since the preprocessed bytes are the same, there should only be on copy
        self.assertEqual(len(program.backend_delegate_data), 1)

        self.check_backend_delegate(
            program=program,
            delegate=program.execution_plan[0].delegates[0],
            expected_id=BackendWithCompilerDemo.__name__,
            expected_processed=b"1version:0#op:demo::aten.sin.default, numel:1, dtype:torch.float32<debug_handle>1#",
        )
        self.check_backend_delegate(
            program=program,
            delegate=program.execution_plan[1].delegates[0],
            expected_id=BackendWithCompilerDemo.__name__,
            expected_processed=b"1version:0#op:demo::aten.sin.default, numel:1, dtype:torch.float32<debug_handle>1#",
        )

        # Check that there are two methods
        self.assertEqual(len(program.execution_plan), 2)

        delegate_method_1 = program.execution_plan[0].delegates
        delegate_method_2 = program.execution_plan[1].delegates

        # 1 delegate blob for each method
        self.assertEqual(len(delegate_method_1), 1)
        self.assertEqual(len(delegate_method_2), 1)

        # Delegate Blobs reference the same underlying bytes
        delegate_reference1 = delegate_method_1[0].processed
        delegate_reference2 = delegate_method_2[0].processed
        self.assertEqual(delegate_reference1.index, delegate_reference2.index)

        et_module = _load_for_executorch_from_buffer(exec_prog.buffer)
        model_inputs = torch.ones(1)
        model_outputs = et_module.run_method("forward", [model_inputs])
        self.assertEqual(model_inputs, torch.ones(1))
        model_outputs_from_copy_method = et_module.run_method(
            "forward_copy", [model_inputs]
        )
        self.assertEqual(model_inputs, torch.ones(1))
        self.assertEqual(model_outputs, model_outputs_from_copy_method)
        self.assertTrue(
            torch.allclose(
                model_outputs[0], 0.8333 * torch.ones(1), atol=1e-03, rtol=1e-03
            )
        )
