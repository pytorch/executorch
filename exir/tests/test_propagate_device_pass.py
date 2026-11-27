# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import operator
import unittest
from copy import deepcopy
from typing import List, NamedTuple, Optional

# Import to register et_copy ops
import executorch.exir.passes._device_copy_ops_registry  # noqa: F401
import torch
from executorch.exir import EdgeCompileConfig, to_edge, to_edge_transform_and_lower
from executorch.exir.backend.backend_api import to_backend
from executorch.exir.backend.compile_spec_schema import CompileSpec
from executorch.exir.backend.partitioner import (
    DelegationSpec,
    Partitioner,
    PartitionResult,
)
from executorch.exir.backend.test.backend_with_compiler_demo import (
    BackendWithCompilerDemo,
)
from executorch.exir.backend.test.device_util import (
    CpuOnlyPartitioner,
    DeviceAwarePartitioner,
)
from executorch.exir.capture._config import ExecutorchBackendConfig
from executorch.exir.delegate import executorch_call_delegate
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportedProgramPassBase, ExportedProgramPassResult
from executorch.exir.passes import (
    MemoryPlanningPass,
    propagate_device_pass,
    ToOutVarPass,
)
from executorch.exir.passes.propagate_device_pass import (
    _get_target_device_from_compile_specs,
    _parse_device_spec_value,
    PropagateDeviceConfig,
    TARGET_DEVICE_COMPILE_SPEC_KEY,
)
from executorch.exir.schema import DeviceType
from executorch.exir.tensor import TensorSpec
from torch.export import export


class DeviceCopyNodes(NamedTuple):
    h2d_nodes: List[torch.fx.Node]
    d2h_nodes: List[torch.fx.Node]
    delegate_nodes: List[torch.fx.Node]
    getitem_nodes: List[torch.fx.Node]


def _lower_model_to_executorch(
    model: torch.nn.Module,
    inputs: tuple,
    partitioner: Partitioner,
    et_config: Optional[ExecutorchBackendConfig] = None,
) -> List:
    """Lower model all the way through to_executorch for E2E tests."""
    if et_config is None:
        et_config = ExecutorchBackendConfig(emit_stacktrace=False)

    ep = export(model, inputs)
    ep_copied = deepcopy(ep)

    edge_1 = to_edge(ep, compile_config=EdgeCompileConfig(_check_ir_validity=False))
    lowered_1 = edge_1.to_backend(partitioner)
    et_1 = lowered_1.to_executorch(deepcopy(et_config))
    gm_1 = et_1.exported_program().graph_module

    edge_2 = to_edge_transform_and_lower(ep_copied, partitioner=[partitioner])
    et_2 = edge_2.to_executorch(deepcopy(et_config))
    gm_2 = et_2.exported_program().graph_module

    return [
        ("to_edge+to_backend", gm_1),
        ("to_edge_transform_and_lower", gm_2),
    ]


def _collect_device_copy_nodes(gm: torch.fx.GraphModule) -> DeviceCopyNodes:
    h2d_nodes = []
    d2h_nodes = []
    delegate_nodes = []
    getitem_nodes = []

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if node.target == torch.ops.et_copy._h2d_copy.out:
            h2d_nodes.append(node)
        elif node.target == torch.ops.et_copy._d2h_copy.out:
            d2h_nodes.append(node)
        elif node.target == executorch_call_delegate:
            delegate_nodes.append(node)
        elif node.target == operator.getitem:
            getitem_nodes.append(node)

    return DeviceCopyNodes(
        h2d_nodes=h2d_nodes,
        d2h_nodes=d2h_nodes,
        delegate_nodes=delegate_nodes,
        getitem_nodes=getitem_nodes,
    )


def _lower_add_to_device(inputs: tuple, device: str) -> torch.nn.Module:
    """Lower a module that returns ``x + y`` to a delegate on ``device``."""

    class Add(torch.nn.Module):
        def forward(self, x, y):
            return torch.add(x, y)

    return to_backend(
        BackendWithCompilerDemo.__name__,
        to_edge(export(Add(), inputs)).exported_program(),
        [
            CompileSpec("max_value", bytes([4])),
            CompileSpec(TARGET_DEVICE_COMPILE_SPEC_KEY, device.encode("utf-8")),
        ],
    )


class PerAddDeviceDelegatePartitioner(Partitioner):
    """Delegates each ``aten.add.Tensor`` as its OWN delegate, each with a
    per-op ``target_device``.

    Unlike ``DeviceAwarePartitioner`` (which merges connected add ops into a
    single delegate), this keeps every add in a separate delegate, so two
    directly-connected adds become two directly-connected delegates -- the shape
    that leaves a redundant ``device -> host -> device`` round-trip between them.
    ``devices[i]`` is the target device for the i-th add in graph order (the last
    entry is reused if there are more adds than devices). ``None`` gives that add
    a delegate without a target device.
    """

    def __init__(self, devices: List[Optional[str]]) -> None:
        super().__init__()
        self._devices = list(devices)

    def partition(self, exported_program) -> PartitionResult:
        partition_tags = {}
        add_nodes = [
            n
            for n in exported_program.graph_module.graph.nodes
            if n.op == "call_function" and n.target == exir_ops.edge.aten.add.Tensor
        ]
        for i, node in enumerate(add_nodes):
            device = self._devices[min(i, len(self._devices) - 1)]
            tag = f"tag{i}"
            node.meta["delegation_tag"] = tag
            compile_specs = [CompileSpec("max_value", bytes([4]))]
            if device is not None:
                compile_specs.append(
                    CompileSpec(TARGET_DEVICE_COMPILE_SPEC_KEY, device.encode("utf-8"))
                )
            partition_tags[tag] = DelegationSpec(
                BackendWithCompilerDemo.__name__, compile_specs
            )
        return PartitionResult(
            tagged_exported_program=exported_program,
            partition_tags=partition_tags,
        )


# A mutable op that stays on the host, like a KV cache update. Once
# to_executorch removes auto_functionalized, it adds other into x in place.
_test_lib = torch.library.Library("propagate_device_test", "DEF")
_test_lib.define("accumulate(Tensor(a!) x, Tensor other) -> Tensor")
_test_lib.define(
    "accumulate.out(Tensor(a!) x, Tensor other, *, Tensor(b!) out) -> Tensor(b!)"
)


@torch.library.impl(_test_lib, "accumulate", "CompositeExplicitAutograd")
def _accumulate(x: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    x.add_(other)
    return x.clone()


@torch.library.impl(_test_lib, "accumulate.out", "CompositeExplicitAutograd")
def _accumulate_out(
    x: torch.Tensor, other: torch.Tensor, out: torch.Tensor
) -> torch.Tensor:
    x.add_(other)
    return out.copy_(x)


def _lower_with_host_writer(
    model: torch.nn.Module,
    inputs: tuple,
    devices: List[str],
    et_config: ExecutorchBackendConfig,
) -> torch.fx.GraphModule:
    """Delegate each add of a model that also calls accumulate on the host."""
    # torch._export.config turns off auto_functionalized v2 for export in some
    # PyTorch builds (see is_fbcode() there). Under v1, a custom op that writes
    # through a view writes a view_copy and the write is lost, a limitation
    # that predates these passes. Pin v2 so these tests check the same graph in
    # every build.
    with torch._export.config.patch(enable_auto_functionalized_v2_for_export=True):
        edge = to_edge(
            export(model, inputs),
            compile_config=EdgeCompileConfig(_check_ir_validity=False),
        )
    et = edge.to_backend(PerAddDeviceDelegatePartitioner(devices)).to_executorch(
        et_config
    )
    return et.exported_program().graph_module


def _host_writer(gm: torch.fx.GraphModule) -> torch.fx.Node:
    (writer,) = [
        n
        for n in gm.graph.nodes
        if n.target == torch.ops.propagate_device_test.accumulate.out
    ]
    return writer


def _host_written(gm: torch.fx.GraphModule) -> torch.fx.Node:
    """The tensor that accumulate writes, looking through the view it may write."""
    written = _host_writer(gm).args[0]
    if written.target == torch.ops.aten.as_strided.default:
        return written.args[0]
    return written


def _host_writer_config(write_view: bool, **kwargs) -> ExecutorchBackendConfig:
    return ExecutorchBackendConfig(
        emit_stacktrace=False,
        enable_non_cpu_memory_planning=True,
        # A write through a view leaves the aten.as_strided that
        # to_executorch makes for it, which has no out variant.
        to_out_var_pass=ToOutVarPass(ignore_to_out_var_failure=write_view),
        **kwargs,
    )


def _skip_h2d_for_constants_config(
    alloc_graph_input: bool, skip_constants: bool = True
) -> ExecutorchBackendConfig:
    return ExecutorchBackendConfig(
        emit_stacktrace=False,
        propagate_device_config=PropagateDeviceConfig(
            skip_h2d_for_method_inputs=True, skip_h2d_for_constants=skip_constants
        ),
        enable_non_cpu_memory_planning=True,
        memory_planning_pass=MemoryPlanningPass(alloc_graph_input=alloc_graph_input),
    )


class TestPropagateDevicePass(unittest.TestCase):
    @staticmethod
    def _collect_tensor_specs(node: torch.fx.Node) -> List[TensorSpec]:
        """Return a flat list of TensorSpecs from a node's 'spec' metadata."""
        spec = node.meta.get("spec")
        if spec is None:
            return []
        if isinstance(spec, TensorSpec):
            return [spec]
        if isinstance(spec, (tuple, list)):
            return [s for s in spec if isinstance(s, TensorSpec)]
        return []

    @staticmethod
    def _is_delegate_getitem(node: torch.fx.Node) -> bool:
        """Return True if *node* is a getitem extracting from a delegate call."""
        if node.target != operator.getitem:
            return False
        source = node.args[0]
        return (
            isinstance(source, torch.fx.Node)
            and source.op == "call_function"
            and source.target == executorch_call_delegate
        )

    def _assert_specs_device(
        self,
        specs: List[TensorSpec],
        expected_device: DeviceType,
        msg: str,
        expected_index: int | None = None,
    ) -> None:
        """Assert every spec has the expected device (and optionally index)."""
        for s in specs:
            self.assertEqual(s.device, expected_device, msg)
            if expected_index is not None:
                self.assertEqual(s.device_index, expected_index)

    def _assert_buffer_device(
        self,
        spec: TensorSpec,
        program,
        expected_device: DeviceType,
        msg: str,
    ) -> None:
        """Assert the emitted program maps the spec's buffer to the expected device.

        The memory planner assigns each TensorSpec a ``mem_id`` (buffer index).
        When ``enable_non_cpu_memory_planning`` is True, non-CPU buffers get an
        entry in ``execution_plan[0].non_const_buffer_device``.  CPU buffers have
        no explicit entry (CPU is the default).
        """
        plan = program.execution_plan[0]
        mem_id = spec.mem_id
        self.assertIsNotNone(mem_id, f"{msg}: spec.mem_id should not be None")

        if expected_device == DeviceType.CPU:
            # CPU buffers have no explicit entry in non_const_buffer_device.
            if plan.non_const_buffer_device is not None:
                for entry in plan.non_const_buffer_device:
                    self.assertNotEqual(
                        entry.buffer_idx,
                        mem_id,
                        f"{msg}: buffer {mem_id} should be CPU but found "
                        f"in non_const_buffer_device as {entry.device_type.name}",
                    )
        else:
            self.assertIsNotNone(
                plan.non_const_buffer_device,
                f"{msg}: non_const_buffer_device should exist for non-CPU buffers",
            )
            matching = [
                e for e in plan.non_const_buffer_device if e.buffer_idx == mem_id
            ]
            self.assertEqual(
                len(matching),
                1,
                f"{msg}: expected exactly one entry for buffer {mem_id} "
                f"in non_const_buffer_device, got {len(matching)}",
            )
            self.assertEqual(
                matching[0].device_type,
                expected_device,
                f"{msg}: buffer {mem_id} device type mismatch",
            )

    @staticmethod
    def _collect_placeholders_by_device(gm):
        """Partition placeholder nodes by device type. Returns (cuda_list, cpu_list)."""
        cuda, cpu = [], []
        for node in gm.graph.nodes:
            if node.op != "placeholder":
                continue
            spec = node.meta.get("spec")
            if isinstance(spec, TensorSpec) and spec.device == DeviceType.CUDA:
                cuda.append(node)
            elif isinstance(spec, TensorSpec):
                cpu.append(node)
        return cuda, cpu

    def _collect_delegate_getitems(self, gm):
        """Return list of getitem nodes extracting from delegate calls."""
        return [n for n in gm.graph.nodes if self._is_delegate_getitem(n)]

    def _assert_nodes_device(
        self, nodes, expected_device, pipeline, label, expected_index=None
    ):
        """Assert every node's TensorSpec has the expected device."""
        for node in nodes:
            spec = node.meta.get("spec")
            if isinstance(spec, TensorSpec):
                self.assertEqual(
                    spec.device,
                    expected_device,
                    f"[{pipeline}] {label} '{node.name}' should have "
                    f"{expected_device.name} device spec",
                )
                if expected_index is not None:
                    self.assertEqual(spec.device_index, expected_index)

    def _assert_nodes_buffer_device(
        self, nodes, program, expected_device, pipeline, label
    ):
        """Assert each node's buffer is mapped to the expected device."""
        for node in nodes:
            spec = node.meta.get("spec")
            if isinstance(spec, TensorSpec):
                self._assert_buffer_device(
                    spec,
                    program,
                    expected_device,
                    f"[{pipeline}] {label} '{node.name}' buffer",
                )

    # ---- Integration tests: copy nodes after to_executorch ----

    def test_h2d_d2h_nodes_inserted(self):
        """Verify H2D/D2H copy nodes are inserted and survive the full
        to_executorch pipeline with correct .out variant targets, exact
        counts, and proper graph ordering."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_to_executorch(
            model,
            inputs,
            DeviceAwarePartitioner("cuda:0"),
            ExecutorchBackendConfig(enable_non_cpu_memory_planning=True),
        ):
            with self.subTest(pipeline=pipeline):
                nodes = _collect_device_copy_nodes(gm)
                h2d_nodes = nodes.h2d_nodes
                d2h_nodes = nodes.d2h_nodes
                delegate_nodes = nodes.delegate_nodes
                getitem_nodes = nodes.getitem_nodes

                # Model has 2 inputs, 1 output → 2 H2D, 1 D2H
                self.assertEqual(
                    len(h2d_nodes),
                    2,
                    f"[{pipeline}] Expected 2 H2D copy nodes (one per "
                    f"delegate input), got {len(h2d_nodes)}",
                )
                self.assertEqual(
                    len(d2h_nodes),
                    1,
                    f"[{pipeline}] Expected 1 D2H copy node (one per "
                    f"delegate output), got {len(d2h_nodes)}",
                )
                self.assertEqual(len(delegate_nodes), 1)

                # Verify graph ordering:
                # placeholder → h2d_copy → delegate → getitem → d2h_copy → output
                all_nodes = list(gm.graph.nodes)
                delegate_idx = all_nodes.index(delegate_nodes[0])
                for h2d in h2d_nodes:
                    self.assertLess(
                        all_nodes.index(h2d),
                        delegate_idx,
                        f"[{pipeline}] H2D '{h2d.name}' must appear before "
                        f"delegate '{delegate_nodes[0].name}'",
                    )
                for d2h in d2h_nodes:
                    for gi in getitem_nodes:
                        if gi.args[0] == delegate_nodes[0]:
                            self.assertGreater(
                                all_nodes.index(d2h),
                                all_nodes.index(gi),
                                f"[{pipeline}] D2H '{d2h.name}' must appear "
                                f"after getitem '{gi.name}'",
                            )

    def test_e2e_copy_nodes_in_executorch_graph(self):
        """End-to-end: copy nodes survive the full to_executorch pipeline
        and have correct .out targets and device specs on TensorSpecs."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_to_executorch(
            model,
            inputs,
            DeviceAwarePartitioner("cuda:0"),
            ExecutorchBackendConfig(enable_non_cpu_memory_planning=True),
        ):
            with self.subTest(pipeline=pipeline):
                nodes = _collect_device_copy_nodes(gm)
                h2d_nodes = nodes.h2d_nodes
                d2h_nodes = nodes.d2h_nodes

                self.assertGreater(
                    len(h2d_nodes),
                    0,
                    f"[{pipeline}] H2D copy nodes must survive to_executorch",
                )
                self.assertGreater(
                    len(d2h_nodes),
                    0,
                    f"[{pipeline}] D2H copy nodes must survive to_executorch",
                )

                for h2d in h2d_nodes:
                    spec = h2d.meta.get("spec")
                    self.assertIsNotNone(
                        spec,
                        f"[{pipeline}] H2D node '{h2d.name}' missing spec",
                    )
                    if isinstance(spec, TensorSpec):
                        self.assertEqual(
                            spec.device,
                            DeviceType.CUDA,
                            f"[{pipeline}] H2D output '{h2d.name}' should be "
                            f"on CUDA, got {spec.device.name}",
                        )
                        self.assertEqual(spec.device_index, 0)

                for d2h in d2h_nodes:
                    spec = d2h.meta.get("spec")
                    self.assertIsNotNone(
                        spec,
                        f"[{pipeline}] D2H node '{d2h.name}' missing spec",
                    )
                    if isinstance(spec, TensorSpec):
                        self.assertEqual(
                            spec.device,
                            DeviceType.CPU,
                            f"[{pipeline}] D2H output '{d2h.name}' should be "
                            f"on CPU, got {spec.device.name}",
                        )

    def test_no_copy_nodes_without_device(self):
        """When the partitioner has no target_device CompileSpec, no H2D/D2H
        copy nodes should be inserted in the final graph."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_to_executorch(
            model, inputs, CpuOnlyPartitioner()
        ):
            with self.subTest(pipeline=pipeline):
                device_copy_nodes = _collect_device_copy_nodes(gm)
                self.assertEqual(
                    len(device_copy_nodes.h2d_nodes),
                    0,
                    f"[{pipeline}] Unexpected H2D copy nodes when no target_device is set",
                )
                self.assertEqual(
                    len(device_copy_nodes.d2h_nodes),
                    0,
                    f"[{pipeline}] Unexpected D2H copy nodes when no target_device is set",
                )

    def test_copy_nodes_require_non_cpu_memory_planning(self):
        """With enable_non_cpu_memory_planning disabled, lowering keeps legacy
        device tags without inserting runtime copy ops."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_to_executorch(
            model,
            inputs,
            DeviceAwarePartitioner("cuda:0"),
            ExecutorchBackendConfig(enable_non_cpu_memory_planning=False),
        ):
            with self.subTest(pipeline=pipeline):
                device_copy_nodes = _collect_device_copy_nodes(gm)
                self.assertEqual(len(device_copy_nodes.h2d_nodes), 0)
                self.assertEqual(len(device_copy_nodes.d2h_nodes), 0)

    def test_no_copy_nodes_for_cpu_target_device(self):
        """A delegate whose target device is the CPU reads and writes host
        memory, so it gets no copies, like a delegate without a target device."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        et_config = ExecutorchBackendConfig(
            emit_stacktrace=False, enable_non_cpu_memory_planning=True
        )
        for pipeline, gm in _lower_model_to_executorch(
            Model(),
            (torch.randn(2, 2), torch.randn(2, 2)),
            PerAddDeviceDelegatePartitioner(["cpu"]),
            et_config,
        ):
            with self.subTest(pipeline=pipeline):
                nodes = _collect_device_copy_nodes(gm)
                self.assertEqual(len(nodes.delegate_nodes), 1)
                self.assertEqual(nodes.h2d_nodes, [])
                self.assertEqual(nodes.d2h_nodes, [])

    def test_redundant_same_device_copies_elided(self):
        """Two directly-connected delegates on the *same* device must not leave a
        redundant device->host->device round-trip between them. The
        inter-delegate _d2h_copy/_h2d_copy pair is elided; only the true
        graph-boundary copies (H2D per input, D2H for the final output) remain."""

        class Model(torch.nn.Module):
            def forward(self, a, b, c):
                return (a + b) + c

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_to_executorch(
            model,
            inputs,
            PerAddDeviceDelegatePartitioner(["cuda:0", "cuda:0"]),
            ExecutorchBackendConfig(enable_non_cpu_memory_planning=True),
        ):
            with self.subTest(pipeline=pipeline):
                nodes = _collect_device_copy_nodes(gm)
                self.assertEqual(
                    len(nodes.delegate_nodes),
                    2,
                    f"[{pipeline}] expected two separate delegates",
                )
                # Boundary copies only: an H2D for each of the 3 graph inputs
                # (a, b, c) and a single D2H for the final output. The first
                # delegate's output feeds the second directly on cuda:0, so the
                # inter-delegate round-trip is elided.
                self.assertEqual(
                    len(nodes.h2d_nodes),
                    3,
                    f"[{pipeline}] expected 3 boundary H2D copies, got "
                    f"{len(nodes.h2d_nodes)}",
                )
                self.assertEqual(
                    len(nodes.d2h_nodes),
                    1,
                    f"[{pipeline}] expected the inter-delegate D2H to be elided "
                    f"(1 boundary D2H), got {len(nodes.d2h_nodes)}",
                )

    def test_same_device_copies_kept_when_host_op_writes_copy(self):
        """A host op that writes the host copy of a delegate output in place,
        directly or through a view, changes what the next delegate on the same
        device reads, so that round-trip stays. The input it only reads still
        gets one copy for both delegates."""

        class Model(torch.nn.Module):
            def __init__(self, write_view):
                super().__init__()
                self.write_view = write_view

            def forward(self, x, y):
                a = x + y
                if self.write_view:
                    torch.ops.propagate_device_test.accumulate(a.view(-1), y.view(-1))
                else:
                    torch.ops.propagate_device_test.accumulate(a, y)
                return a + y

        for write_view in (False, True):
            gm = _lower_with_host_writer(
                Model(write_view),
                (torch.randn(2, 2), torch.randn(2, 2)),
                ["cuda:0", "cuda:0"],
                _host_writer_config(write_view),
            )
            with self.subTest(write_view=write_view):
                written = _host_written(gm)
                self.assertEqual(written.target, torch.ops.et_copy._d2h_copy.out)
                nodes = _collect_device_copy_nodes(gm)
                self.assertIn(
                    written,
                    [
                        arg.args[0]
                        for arg in nodes.delegate_nodes[1].args[1:]
                        if arg.target == torch.ops.et_copy._h2d_copy.out
                    ],
                )
                y_copies = [n for n in nodes.h2d_nodes if n.args[0].name == "y"]
                self.assertEqual(len(y_copies), 1)

    def test_host_writes_after_the_device_readers_keep_no_copies(self):
        """Host ops that write tensors in place only after the delegates on one
        device have read them cannot change what those delegates read. So the
        round-trip between the delegates is still removed, and the tensor both
        delegates read gets one copy."""

        class Model(torch.nn.Module):
            def forward(self, x, y):
                t = x * 2
                a = t + y
                b = a + t
                # Both writes read b, so they run after the second delegate.
                torch.ops.propagate_device_test.accumulate(a, b)
                torch.ops.propagate_device_test.accumulate(t, b)
                return a, b, t

        gm = _lower_with_host_writer(
            Model(),
            (torch.randn(2, 2), torch.randn(2, 2)),
            ["cuda:0", "cuda:0"],
            _host_writer_config(False),
        )
        nodes = _collect_device_copy_nodes(gm)
        first, second = nodes.delegate_nodes
        order = list(gm.graph.nodes)
        writers = [
            n
            for n in order
            if n.target == torch.ops.propagate_device_test.accumulate.out
        ]
        self.assertEqual(len(writers), 2)
        for writer in writers:
            self.assertLess(order.index(second), order.index(writer))
        # Only t and y are copied, and the second delegate reads the first
        # delegate's output directly.
        self.assertEqual(len(nodes.h2d_nodes), 2)
        (shared,) = set(first.args[1:]) & set(second.args[1:])
        self.assertIn(shared, nodes.h2d_nodes)

    def test_single_tensor_delegate_output_read_on_host(self):
        """A host op that reads the output of a lowered module returning one
        tensor reads a host copy of it."""

        inputs = (torch.randn(2, 2), torch.randn(2, 2))
        lowered = _lower_add_to_device(inputs, "cuda:0")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered = lowered

            def forward(self, x, y):
                return torch.sin(self.lowered(x, y))

        et = to_edge(export(Model(), inputs)).to_executorch(
            ExecutorchBackendConfig(
                emit_stacktrace=False, enable_non_cpu_memory_planning=True
            )
        )
        gm = et.exported_program().graph_module
        nodes = _collect_device_copy_nodes(gm)
        (d2h,) = nodes.d2h_nodes
        self.assertIs(d2h.args[0], nodes.delegate_nodes[0])
        (sin,) = d2h.users
        self.assertEqual(sin.target, torch.ops.aten.sin.out)
        self._assert_nodes_buffer_device(
            [d2h], et._emitter_output.program, DeviceType.CPU, "host op", "D2H copy"
        )

    def test_single_tensor_delegate_output_copied_to_host(self):
        """A lowered module that returns one tensor, called from another module,
        has no getitem after its call. Its device output still gets a _d2h_copy,
        and the next call on the same device reads it without a round-trip. With
        skip_h2d_for_method_inputs, y, read by both calls, is placed once."""

        inputs = (torch.randn(2, 2), torch.randn(2, 2))
        lowered = _lower_add_to_device(inputs, "cuda:0")

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered = lowered

            def forward(self, x, y):
                return self.lowered(self.lowered(x, y), y)

        configs = (
            PropagateDeviceConfig(),
            PropagateDeviceConfig(skip_h2d_for_method_inputs=True),
            PropagateDeviceConfig(skip_d2h_for_method_outputs=True),
        )
        for device_config in configs:
            skip_h2d = device_config.skip_h2d_for_method_inputs
            skip_d2h = device_config.skip_d2h_for_method_outputs
            et = to_edge(export(Model(), inputs)).to_executorch(
                ExecutorchBackendConfig(
                    emit_stacktrace=False,
                    propagate_device_config=device_config,
                    enable_non_cpu_memory_planning=True,
                )
            )
            program = et._emitter_output.program
            gm = et.exported_program().graph_module
            case = f"skip_h2d={skip_h2d}, skip_d2h={skip_d2h}"
            with self.subTest(case=case):
                nodes = _collect_device_copy_nodes(gm)
                self.assertEqual(len(nodes.delegate_nodes), 2)
                # Only method inputs are copied to the device, x and y once
                # each, unless the caller places them.
                self.assertEqual(len(nodes.h2d_nodes), 0 if skip_h2d else 2)
                self.assertTrue(
                    all(n.args[0].op == "placeholder" for n in nodes.h2d_nodes)
                )
                cuda_ph, _ = self._collect_placeholders_by_device(gm)
                self.assertEqual(len(cuda_ph), 2 if skip_h2d else 0)
                # Only the method output can be copied back to the host.
                self.assertEqual(len(nodes.d2h_nodes), 0 if skip_d2h else 1)
                (output,) = gm.graph.output_node().args[0]
                self.assertEqual(
                    output.target,
                    (
                        executorch_call_delegate
                        if skip_d2h
                        else torch.ops.et_copy._d2h_copy.out
                    ),
                )
                self._assert_nodes_buffer_device(
                    nodes.delegate_nodes, program, DeviceType.CUDA, case, "Delegate"
                )
                self._assert_nodes_buffer_device(
                    nodes.d2h_nodes, program, DeviceType.CPU, case, "D2H copy"
                )

    def test_host_inplace_ops_on_delegate_output_write_host_copy(self):
        """Host ops that write a delegate output in place, here relu_ and then
        mul_ from reinplace_pass, write its host copy. Their results share the
        copy's spec, as memory planning requires, and the method returns the
        last one from the host."""

        inputs = (torch.randn(2, 2), torch.randn(2, 2))
        lowered = _lower_add_to_device(inputs, "cuda:0")

        class Call(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered = lowered

            def forward(self, x, y):
                return torch.relu(self.lowered(x, y)) * 2

        class Add(torch.nn.Module):
            def forward(self, x, y):
                return torch.relu(x + y) * 2

        cases = {
            "single tensor": (Call(), None),
            "getitem": (Add(), PerAddDeviceDelegatePartitioner(["cuda:0"])),
        }
        reinplaced = frozenset(
            {exir_ops.edge.aten.relu.default, exir_ops.edge.aten.mul.Tensor}
        )
        for case, (model, partitioner) in cases.items():
            for skip_d2h in (False, True):
                edge = to_edge(export(model, inputs))
                if partitioner is not None:
                    edge = edge.to_backend(partitioner)
                et = edge.to_executorch(
                    ExecutorchBackendConfig(
                        emit_stacktrace=False,
                        reinplace_extra_ops=reinplaced,
                        propagate_device_config=PropagateDeviceConfig(
                            skip_d2h_for_method_outputs=skip_d2h
                        ),
                        enable_non_cpu_memory_planning=True,
                    )
                )
                gm = et.exported_program().graph_module
                with self.subTest(case=case, skip_d2h=skip_d2h):
                    (d2h,) = _collect_device_copy_nodes(gm).d2h_nodes
                    (relu,) = d2h.users
                    (mul,) = relu.users
                    self.assertEqual(relu.target, exir_ops.edge.aten.relu_.default)
                    self.assertEqual(mul.target, exir_ops.edge.aten.mul_.Tensor)
                    self.assertIs(relu.meta["spec"], d2h.meta["spec"])
                    self.assertIs(mul.meta["spec"], d2h.meta["spec"])
                    self.assertEqual(gm.graph.output_node().args[0], (mul,))
                    self._assert_nodes_buffer_device(
                        [mul],
                        et._emitter_output.program,
                        DeviceType.CPU,
                        case,
                        "Output",
                    )

    def test_skip_d2h_keeps_output_also_read_on_device(self):
        """With skip_d2h_for_method_outputs, a returned delegate output that a
        delegate on the same device also reads stays on the device, like the
        other outputs."""

        inputs = (torch.randn(2, 2), torch.randn(2, 2))
        lowered = _lower_add_to_device(inputs, "cuda:0")

        class Calls(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered = lowered

            def forward(self, x, y):
                a = self.lowered(x, y)
                return a, self.lowered(a, y)

        class Adds(torch.nn.Module):
            def forward(self, x, y):
                a = torch.add(x, y)
                return a, torch.add(a, y)

        et_config = ExecutorchBackendConfig(
            emit_stacktrace=False,
            propagate_device_config=PropagateDeviceConfig(
                skip_d2h_for_method_outputs=True
            ),
            enable_non_cpu_memory_planning=True,
        )
        cases = {
            "single tensor": (Calls(), None),
            "getitem": (Adds(), PerAddDeviceDelegatePartitioner(["cuda:0"])),
        }
        for case, (model, partitioner) in cases.items():
            edge = to_edge(export(model, inputs))
            if partitioner is not None:
                edge = edge.to_backend(partitioner)
            et = edge.to_executorch(deepcopy(et_config))
            gm = et.exported_program().graph_module
            with self.subTest(case=case):
                self.assertEqual(len(_collect_device_copy_nodes(gm).d2h_nodes), 0)
                outputs = gm.graph.output_node().args[0]
                self._assert_nodes_device(outputs, DeviceType.CUDA, case, "Output")
                self._assert_nodes_buffer_device(
                    outputs, et._emitter_output.program, DeviceType.CUDA, case, "Output"
                )

    def test_skip_d2h_keeps_output_also_read_off_device(self):
        """With skip_d2h_for_method_outputs, a returned delegate output that a
        host op or a delegate on another device also reads stays on the device.
        Only that other reader gets the host copy."""

        inputs = (torch.randn(2, 2), torch.randn(2, 2))
        lowered = _lower_add_to_device(inputs, "cuda:0")
        lowered_cuda_1 = _lower_add_to_device(inputs, "cuda:1")

        class HostReader(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered = lowered

            def forward(self, x, y):
                a = self.lowered(x, y)
                return a, torch.sin(a)

        class OtherDeviceReader(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lowered = lowered
                self.lowered_cuda_1 = lowered_cuda_1

            def forward(self, x, y):
                a = self.lowered(x, y)
                return a, self.lowered_cuda_1(a, y)

        et_config = ExecutorchBackendConfig(
            emit_stacktrace=False,
            propagate_device_config=PropagateDeviceConfig(
                skip_d2h_for_method_outputs=True
            ),
            enable_non_cpu_memory_planning=True,
        )
        for case, model in {
            "host op": HostReader(),
            "delegate on cuda:1": OtherDeviceReader(),
        }.items():
            et = to_edge(export(model, inputs)).to_executorch(deepcopy(et_config))
            gm = et.exported_program().graph_module
            with self.subTest(case=case):
                output = gm.graph.output_node().args[0][0]
                self.assertEqual(output.target, executorch_call_delegate)
                self._assert_nodes_device([output], DeviceType.CUDA, case, "Output", 0)
                self._assert_nodes_buffer_device(
                    [output],
                    et._emitter_output.program,
                    DeviceType.CUDA,
                    case,
                    "Output",
                )
                (d2h,) = _collect_device_copy_nodes(gm).d2h_nodes
                self.assertIs(d2h.args[0], output)
                self.assertNotIn(gm.graph.output_node(), d2h.users)

    def test_skip_d2h_leaves_returned_output_unplanned(self):
        """With skip_d2h_for_method_outputs and unplanned graph outputs, the
        caller provides the returned delegate output, while the host copy that
        a host op reads is still planned."""

        class Model(torch.nn.Module):
            def forward(self, x, y):
                a = x + y
                return a, torch.sin(a)

        et = (
            to_edge(export(Model(), (torch.randn(2, 2), torch.randn(2, 2))))
            .to_backend(PerAddDeviceDelegatePartitioner(["cuda:0"]))
            .to_executorch(
                ExecutorchBackendConfig(
                    emit_stacktrace=False,
                    propagate_device_config=PropagateDeviceConfig(
                        skip_d2h_for_method_outputs=True
                    ),
                    enable_non_cpu_memory_planning=True,
                    memory_planning_pass=MemoryPlanningPass(alloc_graph_output=False),
                )
            )
        )
        gm = et.exported_program().graph_module
        output = gm.graph.output_node().args[0][0]
        (d2h,) = _collect_device_copy_nodes(gm).d2h_nodes
        self.assertIs(d2h.args[0], output)
        self.assertIsNone(output.meta["spec"].mem_id)
        self.assertIsNotNone(d2h.meta["spec"].mem_id)

    def test_skip_d2h_returns_output_written_on_host_from_host(self):
        """With skip_d2h_for_method_outputs, a returned delegate output whose
        host copy a host op writes in place, directly or through a view, is
        returned as that written host copy, because the device tensor never
        sees the write. No copy goes back to the device. A warning names the
        op that writes it, since the copy to the host runs on every call."""

        class Model(torch.nn.Module):
            def __init__(self, write_view):
                super().__init__()
                self.write_view = write_view

            def forward(self, x, y):
                a = x + y
                if self.write_view:
                    torch.ops.propagate_device_test.accumulate(a.view(-1), y.view(-1))
                else:
                    torch.ops.propagate_device_test.accumulate(a, y)
                return a

        for write_view in (False, True):
            with self.assertLogs(propagate_device_pass.logger, "WARNING") as logs:
                gm = _lower_with_host_writer(
                    Model(write_view),
                    (torch.randn(2, 2), torch.randn(2, 2)),
                    ["cuda:0"],
                    _host_writer_config(
                        write_view,
                        propagate_device_config=PropagateDeviceConfig(
                            skip_d2h_for_method_outputs=True
                        ),
                    ),
                )
            with self.subTest(write_view=write_view):
                (warning,) = logs.output
                self.assertIn("Method output 0 (16 bytes)", warning)
                self.assertIn(
                    "'propagate_device_test.accumulate.default' writes it", warning
                )
                (output,) = gm.graph.output_node().args[0]
                self.assertIs(output, _host_written(gm))
                self.assertEqual(output.target, torch.ops.et_copy._d2h_copy.out)
                (output_spec,) = gm.graph.output_node().meta["spec"]
                self.assertEqual(output_spec.device, DeviceType.CPU)
                self.assertFalse(
                    any(
                        copy.args[0] is output
                        for copy in _collect_device_copy_nodes(gm).h2d_nodes
                    )
                )

    def test_skip_d2h_copies_a_written_output_only_for_a_later_delegate(self):
        """With skip_d2h_for_method_outputs, a returned output that a host op
        writes before a later delegate on the same device reads it is copied
        back to the device for that delegate only. The method returns the
        written host copy."""

        class Model(torch.nn.Module):
            def forward(self, x, y):
                a = x + y
                torch.ops.propagate_device_test.accumulate(a, y)
                return a, a + y

        gm = _lower_with_host_writer(
            Model(),
            (torch.randn(2, 2), torch.randn(2, 2)),
            ["cuda:0", "cuda:0"],
            _host_writer_config(
                False,
                propagate_device_config=PropagateDeviceConfig(
                    skip_d2h_for_method_outputs=True
                ),
            ),
        )
        (copy,) = [
            n
            for n in _collect_device_copy_nodes(gm).h2d_nodes
            if n.args[0] is _host_written(gm)
        ]
        _, second = _collect_device_copy_nodes(gm).delegate_nodes
        self.assertIn(copy, second.args[1:])
        self.assertIs(gm.graph.output_node().args[0][0], _host_written(gm))

    def test_cross_device_copies_preserved(self):
        """A genuine cross-device hand-off (cuda:0 -> cuda:1) between two
        directly-connected delegates must be preserved -- the round-trip is a
        real cuda:0->host->cuda:1 move, not a redundant same-device bounce."""

        class Model(torch.nn.Module):
            def forward(self, a, b, c):
                return (a + b) + c

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_to_executorch(
            model,
            inputs,
            PerAddDeviceDelegatePartitioner(["cuda:0", "cuda:1"]),
            ExecutorchBackendConfig(enable_non_cpu_memory_planning=True),
        ):
            with self.subTest(pipeline=pipeline):
                nodes = _collect_device_copy_nodes(gm)
                self.assertEqual(len(nodes.delegate_nodes), 2)
                # The inter-delegate D2H (cuda:0->host) and H2D (host->cuda:1)
                # are kept: 4 H2D (a, b on cuda:0; first-add output + c on
                # cuda:1) and 2 D2H (first delegate output, final output).
                self.assertEqual(
                    len(nodes.h2d_nodes),
                    4,
                    f"[{pipeline}] cross-device H2D copies should be preserved, "
                    f"got {len(nodes.h2d_nodes)}",
                )
                self.assertEqual(
                    len(nodes.d2h_nodes),
                    2,
                    f"[{pipeline}] cross-device D2H copies should be preserved, "
                    f"got {len(nodes.d2h_nodes)}",
                )

    def test_output_feeding_multiple_same_device_delegates_elided(self):
        """A delegate output consumed by *multiple* same-device delegates must not
        leave a redundant round-trip on any of those edges.

        The first add's output feeds two separate cuda:0 delegates.
        PropagateDevicePass inserts a single D2H after it whose result is wrapped
        by one H2D per consuming delegate; every such same-device round-trip must
        be elided, and the shared D2H dropped once all its consumers are rewired.
        Only the graph-input H2D copies and the single graph-output D2H remain."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                t = a + b  # delegate 0
                u = t + a  # delegate 1 -- consumes delegate 0's output
                w = t + b  # delegate 2 -- also consumes delegate 0's output
                return u + w  # delegate 3

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_to_executorch(
            model,
            inputs,
            PerAddDeviceDelegatePartitioner(["cuda:0", "cuda:0", "cuda:0", "cuda:0"]),
            ExecutorchBackendConfig(enable_non_cpu_memory_planning=True),
        ):
            with self.subTest(pipeline=pipeline):
                nodes = _collect_device_copy_nodes(gm)
                self.assertEqual(
                    len(nodes.delegate_nodes),
                    4,
                    f"[{pipeline}] expected four delegates",
                )
                # a -> {D0, D1} and b -> {D0, D2} give one graph-input H2D copy
                # each, shared by the delegates that read it; the single graph
                # output gives 1 D2H. All inter-delegate round-trips (including
                # delegate 0's output feeding D1 and D2) are elided.
                self.assertEqual(
                    len(nodes.h2d_nodes),
                    2,
                    f"[{pipeline}] expected 2 boundary H2D copies, got "
                    f"{len(nodes.h2d_nodes)}",
                )
                self.assertEqual(
                    len(nodes.d2h_nodes),
                    1,
                    f"[{pipeline}] expected 1 boundary D2H copy, got "
                    f"{len(nodes.d2h_nodes)}",
                )
                # No H2D may still wrap a D2H (a device->host->device round-trip
                # on the same device), including the shared D2H.
                for h2d in nodes.h2d_nodes:
                    src = h2d.args[0]
                    self.assertFalse(
                        isinstance(src, torch.fx.Node)
                        and src.target == torch.ops.et_copy._d2h_copy.out,
                        f"[{pipeline}] residual same-device round-trip via "
                        f"{h2d.name}",
                    )

    def test_output_feeding_gpu_and_cpu_consumers(self):
        """A delegate output feeding one same-device delegate AND one
        non-delegated (CPU) consumer.

        The GPU->GPU edge is elided -- the second delegate takes the first
        delegate's output directly -- while the D2H that feeds the CPU consumer is
        kept (its shared D2H is still live, so eliminate_dead_code preserves it)."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                t = a + b  # delegate 0 (cuda:0)
                u = t + a  # delegate 1 (cuda:0) -- GPU consumer of t
                w = torch.relu(t)  # non-delegated CPU consumer of t
                return u, w

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        d2h_out = torch.ops.et_copy._d2h_copy.out
        for pipeline, gm in _lower_model_to_executorch(
            model,
            inputs,
            PerAddDeviceDelegatePartitioner(["cuda:0", "cuda:0"]),
            ExecutorchBackendConfig(enable_non_cpu_memory_planning=True),
        ):
            with self.subTest(pipeline=pipeline):
                nodes = _collect_device_copy_nodes(gm)
                self.assertEqual(
                    len(nodes.delegate_nodes),
                    2,
                    f"[{pipeline}] expected two delegates (relu stays on CPU)",
                )
                # GPU->GPU edge elided: no H2D still wraps a D2H.
                for h2d in nodes.h2d_nodes:
                    src = h2d.args[0]
                    self.assertFalse(
                        isinstance(src, torch.fx.Node) and src.target == d2h_out,
                        f"[{pipeline}] GPU->GPU round-trip not elided via {h2d.name}",
                    )
                # The second delegate consumes the first delegate's output
                # directly (a delegate getitem is a direct delegate input).
                direct = any(
                    self._is_delegate_getitem(arg)
                    for d in nodes.delegate_nodes
                    for arg in d.args[1:]
                    if isinstance(arg, torch.fx.Node)
                )
                self.assertTrue(
                    direct,
                    f"[{pipeline}] a delegate should consume another delegate's "
                    f"output directly on device",
                )
                # The D2H feeding the CPU (relu) consumer must be kept.
                relu_fed_by_d2h = any(
                    "relu" in str(user.target)
                    for dn in nodes.d2h_nodes
                    for user in dn.users
                )
                self.assertTrue(
                    relu_fed_by_d2h,
                    f"[{pipeline}] the D2H feeding the CPU consumer must be kept",
                )

    def test_output_feeding_same_and_cross_device_delegates(self):
        """A delegate output feeding two delegates -- one on the SAME device and
        one on a DIFFERENT device.

        The same-device edge is elided (that delegate takes the output directly),
        while the cross-device edge keeps its device->host->device round-trip. The
        shared D2H stays live for the cross-device consumer, so it is not dropped;
        only the genuinely-redundant same-device H2D is removed."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                t = a + b  # delegate 0 (cuda:0)
                u = t + a  # delegate 1 (cuda:0) -- same-device consumer, elided
                w = t + b  # delegate 2 (cuda:1) -- cross-device consumer, kept
                return u, w

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        d2h_out = torch.ops.et_copy._d2h_copy.out
        for pipeline, gm in _lower_model_to_executorch(
            model,
            inputs,
            PerAddDeviceDelegatePartitioner(["cuda:0", "cuda:0", "cuda:1"]),
            ExecutorchBackendConfig(enable_non_cpu_memory_planning=True),
        ):
            with self.subTest(pipeline=pipeline):
                nodes = _collect_device_copy_nodes(gm)
                self.assertEqual(
                    len(nodes.delegate_nodes),
                    3,
                    f"[{pipeline}] expected three delegates",
                )
                # Exactly the cross-device edge keeps its round-trip: one H2D
                # still wraps a D2H, and it targets the other device (cuda:1).
                residual = [
                    h2d
                    for h2d in nodes.h2d_nodes
                    if isinstance(h2d.args[0], torch.fx.Node)
                    and h2d.args[0].target == d2h_out
                ]
                self.assertEqual(
                    len(residual),
                    1,
                    f"[{pipeline}] expected exactly one preserved cross-device "
                    f"round-trip, got {len(residual)}",
                )
                spec = residual[0].meta.get("spec")
                self.assertIsInstance(spec, TensorSpec)
                self.assertEqual(spec.device, DeviceType.CUDA)
                self.assertEqual(
                    spec.device_index,
                    1,
                    f"[{pipeline}] preserved round-trip should target cuda:1",
                )
                # The same-device consumer takes the first delegate's output
                # directly on device.
                direct = any(
                    self._is_delegate_getitem(arg)
                    for d in nodes.delegate_nodes
                    for arg in d.args[1:]
                    if isinstance(arg, torch.fx.Node)
                )
                self.assertTrue(
                    direct,
                    f"[{pipeline}] the same-device delegate should consume the "
                    f"first delegate's output directly",
                )

    def test_tensors_shared_by_delegates_on_different_devices(self):
        """Delegates on cuda:0 and cuda:1 that read the same method input and
        each other's outputs. The input gets one copy for each device, read by
        every delegate there, and each output crosses to the other device
        through the host."""

        class Model(torch.nn.Module):
            def forward(self, x, y):
                a = x + y  # delegate 0 (cuda:0)
                b = x + a  # delegate 1 (cuda:1)
                return a, b, x + b  # delegate 2 (cuda:0)

        d2h_out = torch.ops.et_copy._d2h_copy.out
        for pipeline, gm in _lower_model_to_executorch(
            Model(),
            (torch.randn(2, 2), torch.randn(2, 2)),
            PerAddDeviceDelegatePartitioner(["cuda:0", "cuda:1", "cuda:0"]),
            ExecutorchBackendConfig(enable_non_cpu_memory_planning=True),
        ):
            with self.subTest(pipeline=pipeline):
                nodes = _collect_device_copy_nodes(gm)
                first, second, third = nodes.delegate_nodes
                x_copies = [n for n in nodes.h2d_nodes if n.args[0].name == "x"]
                first_x, second_x, third_x = (
                    next(arg for arg in delegate.args[1:] if arg in x_copies)
                    for delegate in (first, second, third)
                )
                self.assertEqual(len(x_copies), 2)
                self.assertIs(first_x, third_x)
                self.assertEqual(first_x.meta["spec"].device_index, 0)
                self.assertEqual(second_x.meta["spec"].device_index, 1)
                crossings = sorted(
                    (
                        h2d.args[0].args[0].meta["spec"].device_index,
                        h2d.meta["spec"].device_index,
                    )
                    for h2d in nodes.h2d_nodes
                    if h2d.args[0].target == d2h_out
                )
                self.assertEqual(crossings, [(0, 1), (1, 0)])

        # ---- Integration tests: device consistency after to_executorch ----

    def test_device_consistency_cuda_1(self):
        """Verify device tags are correct with cuda:1 after to_executorch()
        to verify device_index propagation through the full pipeline."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_to_executorch(
            model, inputs, DeviceAwarePartitioner("cuda:1")
        ):
            with self.subTest(pipeline=pipeline):
                for node in gm.graph.nodes:
                    if node.op != "call_function":
                        continue
                    specs = self._collect_tensor_specs(node)
                    if not specs:
                        continue

                    label = f"[{pipeline}] '{node.name}'"
                    if node.target == executorch_call_delegate:
                        self._assert_specs_device(
                            specs,
                            DeviceType.CUDA,
                            f"{label} Delegate should be CUDA",
                            expected_index=1,
                        )
                    elif self._is_delegate_getitem(node):
                        self._assert_specs_device(
                            specs,
                            DeviceType.CUDA,
                            f"{label} Delegate getitem should be CUDA",
                            expected_index=1,
                        )

    def test_no_device_spec_remains_cpu(self):
        """When partitioner has no target_device, all specs remain CPU
        through the full to_executorch pipeline."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_to_executorch(
            model, inputs, CpuOnlyPartitioner()
        ):
            with self.subTest(pipeline=pipeline):
                for node in gm.graph.nodes:
                    specs = self._collect_tensor_specs(node)
                    for s in specs:
                        self.assertEqual(
                            s.device,
                            DeviceType.CPU,
                            f"[{pipeline}] All specs should be CPU when no "
                            f"target_device, but node '{node.name}' is {s.device.name}",
                        )

    def test_device_consistency_after_to_executorch(self):
        """Verify device tags are correct in the final graph after
        to_executorch(), not just after PropagateDevicePass alone.
        Copy nodes should bridge CPU ↔ device at delegate boundaries."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, gm in _lower_model_to_executorch(
            model,
            inputs,
            DeviceAwarePartitioner("cuda:0"),
            ExecutorchBackendConfig(enable_non_cpu_memory_planning=True),
        ):
            with self.subTest(pipeline=pipeline):
                for node in gm.graph.nodes:
                    if node.op != "call_function":
                        continue
                    specs = self._collect_tensor_specs(node)
                    if not specs:
                        continue

                    label = f"[{pipeline}] '{node.name}'"
                    if node.target == torch.ops.et_copy._h2d_copy.out:
                        self._assert_specs_device(
                            specs,
                            DeviceType.CUDA,
                            f"{label} H2D output should be CUDA",
                            expected_index=0,
                        )
                    elif node.target == torch.ops.et_copy._d2h_copy.out:
                        self._assert_specs_device(
                            specs,
                            DeviceType.CPU,
                            f"{label} D2H output should be CPU",
                        )
                    elif node.target == executorch_call_delegate:
                        self._assert_specs_device(
                            specs,
                            DeviceType.CUDA,
                            f"{label} Delegate should be CUDA",
                            expected_index=0,
                        )
                    elif self._is_delegate_getitem(node):
                        self._assert_specs_device(
                            specs,
                            DeviceType.CUDA,
                            f"{label} Delegate getitem should be CUDA",
                            expected_index=0,
                        )

    # --- Unit tests for helper functions ---

    def test_parse_device_spec_value(self):
        dt, idx = _parse_device_spec_value(b"cuda:0")
        self.assertEqual(dt, DeviceType.CUDA)
        self.assertEqual(idx, 0)

        dt, idx = _parse_device_spec_value(b"cuda:1")
        self.assertEqual(dt, DeviceType.CUDA)
        self.assertEqual(idx, 1)

        dt, idx = _parse_device_spec_value(b"cpu")
        self.assertEqual(dt, DeviceType.CPU)
        self.assertEqual(idx, 0)

    def test_parse_device_spec_value_unknown_raises(self):
        with self.assertRaises(ValueError):
            _parse_device_spec_value(b"tpu:0")

    def test_parse_device_spec_value_case_insensitive(self):
        dt, idx = _parse_device_spec_value(b"CUDA:0")
        self.assertEqual(dt, DeviceType.CUDA)
        self.assertEqual(idx, 0)

        dt, idx = _parse_device_spec_value(b"Cuda:2")
        self.assertEqual(dt, DeviceType.CUDA)
        self.assertEqual(idx, 2)

    def test_get_target_device_from_compile_specs(self):
        class MockLoweredModule:
            __slots__ = ["compile_specs"]

            def __init__(self, specs):
                self.compile_specs = specs

        module_with_cuda = MockLoweredModule(
            [
                CompileSpec("max_value", bytes([4])),
                CompileSpec(TARGET_DEVICE_COMPILE_SPEC_KEY, b"cuda:0"),
            ]
        )
        result = _get_target_device_from_compile_specs(module_with_cuda)
        self.assertIsNotNone(result)
        dt, idx = result
        self.assertEqual(dt, DeviceType.CUDA)
        self.assertEqual(idx, 0)

        module_without_device = MockLoweredModule(
            [
                CompileSpec("max_value", bytes([4])),
            ]
        )
        result = _get_target_device_from_compile_specs(module_without_device)
        self.assertIsNone(result)

    # ---- End-to-end tests: verify device info survives to_executorch ----

    def _get_executorch_program(self, model, inputs, partitioner, et_config=None):
        """Run the full pipeline and return (emitted_program, graph_module) pairs
        for both export pipelines."""
        if et_config is None:
            et_config = ExecutorchBackendConfig(emit_stacktrace=False)

        ep = export(model, inputs)
        ep_copied = deepcopy(ep)

        # Pipeline 1: to_edge → to_backend → to_executorch
        edge_1 = to_edge(ep, compile_config=EdgeCompileConfig(_check_ir_validity=False))
        lowered_1 = edge_1.to_backend(partitioner)
        et_1 = lowered_1.to_executorch(deepcopy(et_config))
        program_1 = et_1._emitter_output.program
        gm_1 = et_1.exported_program().graph_module

        # Pipeline 2: to_edge_transform_and_lower → to_executorch
        edge_2 = to_edge_transform_and_lower(ep_copied, partitioner=[partitioner])
        et_2 = edge_2.to_executorch(deepcopy(et_config))
        program_2 = et_2._emitter_output.program
        gm_2 = et_2.exported_program().graph_module

        return [
            ("to_edge+to_backend", program_1, gm_1),
            ("to_edge_transform_and_lower", program_2, gm_2),
        ]

    def test_e2e_device_on_specs_after_to_executorch(self):
        """
        After the full to_executorch pipeline, delegate output TensorSpecs
        should still have device == CUDA on the graph module nodes.
        """

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, _program, gm in self._get_executorch_program(
            model, inputs, DeviceAwarePartitioner("cuda:0")
        ):
            with self.subTest(pipeline=pipeline):
                found_delegate = False
                for node in gm.graph.nodes:
                    if (
                        node.op == "call_function"
                        and node.target == executorch_call_delegate
                    ):
                        found_delegate = True
                        specs = node.meta.get("spec")
                        self.assertIsNotNone(specs)
                        if isinstance(specs, TensorSpec):
                            self.assertEqual(
                                specs.device,
                                DeviceType.CUDA,
                                f"[{pipeline}] spec.device should be CUDA after to_executorch",
                            )
                        elif isinstance(specs, (tuple, list)):
                            for s in specs:
                                if isinstance(s, TensorSpec):
                                    self.assertEqual(
                                        s.device,
                                        DeviceType.CUDA,
                                        f"[{pipeline}] spec.device should be CUDA after to_executorch",
                                    )

                self.assertTrue(found_delegate)

    def test_e2e_non_delegated_tensor_specs_remain_cpu(self):
        """
        After to_executorch, non-delegated node specs should still be CPU.
        Getitem nodes extracting from a delegate call are considered delegated.
        """

        class Model(torch.nn.Module):
            def forward(self, a, b):
                c = torch.add(a, b)
                d = torch.sin(c)
                return d

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))

        for pipeline, _program, gm in self._get_executorch_program(
            model, inputs, DeviceAwarePartitioner("cuda:0")
        ):
            with self.subTest(pipeline=pipeline):
                for node in gm.graph.nodes:
                    if node.op != "call_function":
                        continue
                    # Skip delegate call nodes
                    if node.target == executorch_call_delegate:
                        continue
                    # Skip getitem nodes that extract from a delegate call
                    if node.target == operator.getitem:
                        source = node.args[0]
                        if (
                            isinstance(source, torch.fx.Node)
                            and source.op == "call_function"
                            and source.target == executorch_call_delegate
                        ):
                            continue

    # ---- Skip-copy optimization tests ----

    def test_skip_h2d_for_method_inputs(self):
        """When skip_h2d_for_method_inputs=True, placeholder inputs feeding
        directly into a device delegate should NOT get _h2d_copy nodes."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))
        et_config = ExecutorchBackendConfig(
            emit_stacktrace=False,
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True
            ),
            enable_non_cpu_memory_planning=True,
        )

        for pipeline, program, gm in self._get_executorch_program(
            model, inputs, DeviceAwarePartitioner("cuda:0"), et_config
        ):
            with self.subTest(pipeline=pipeline):
                nodes = _collect_device_copy_nodes(gm)
                self.assertEqual(
                    len(nodes.h2d_nodes),
                    0,
                    f"[{pipeline}] Expected no H2D copy nodes when "
                    f"skip_h2d_for_method_inputs=True, got {len(nodes.h2d_nodes)}",
                )
                self.assertEqual(
                    len(nodes.d2h_nodes),
                    1,
                    f"[{pipeline}] Expected 1 D2H copy node for the single "
                    f"output, got {len(nodes.d2h_nodes)}",
                )

                # Placeholder inputs should be tagged as CUDA since H2D was
                # skipped and the pass sets their spec to the target device.
                cuda_ph, cpu_ph = self._collect_placeholders_by_device(gm)
                self.assertEqual(len(cpu_ph), 0)
                self._assert_nodes_device(
                    cuda_ph,
                    DeviceType.CUDA,
                    pipeline,
                    "Placeholder",
                    expected_index=0,
                )

                # Verify buffer device mapping: CUDA placeholders should
                # have their memory planned on a CUDA buffer.
                self._assert_nodes_buffer_device(
                    cuda_ph,
                    program,
                    DeviceType.CUDA,
                    pipeline,
                    "Placeholder",
                )

    def test_skip_d2h_for_method_outputs(self):
        """When skip_d2h_for_method_outputs=True, delegate outputs that feed
        directly to the graph output should NOT get _d2h_copy nodes."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))
        et_config = ExecutorchBackendConfig(
            emit_stacktrace=False,
            propagate_device_config=PropagateDeviceConfig(
                skip_d2h_for_method_outputs=True
            ),
            enable_non_cpu_memory_planning=True,
        )

        for pipeline, program, gm in self._get_executorch_program(
            model, inputs, DeviceAwarePartitioner("cuda:0"), et_config
        ):
            with self.subTest(pipeline=pipeline):
                nodes = _collect_device_copy_nodes(gm)
                self.assertEqual(
                    len(nodes.d2h_nodes),
                    0,
                    f"[{pipeline}] Expected no D2H copy nodes when "
                    f"skip_d2h_for_method_outputs=True, got {len(nodes.d2h_nodes)}",
                )
                self.assertEqual(
                    len(nodes.h2d_nodes),
                    2,
                    f"[{pipeline}] Expected 2 H2D copy nodes for the two "
                    f"inputs, got {len(nodes.h2d_nodes)}",
                )

                # Delegate getitem nodes feeding to output should stay on
                # CUDA since D2H was skipped.
                getitems = self._collect_delegate_getitems(gm)
                self._assert_nodes_device(
                    getitems,
                    DeviceType.CUDA,
                    pipeline,
                    "Delegate getitem",
                )

                # Verify buffer device mapping: CUDA getitem outputs should
                # have their memory planned on a CUDA buffer.
                self._assert_nodes_buffer_device(
                    getitems,
                    program,
                    DeviceType.CUDA,
                    pipeline,
                    "Getitem",
                )

    def test_skip_both_h2d_and_d2h(self):
        """When both skip flags are True, neither H2D nor D2H copy nodes
        should be inserted for a direct input->delegate->output flow."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))
        et_config = ExecutorchBackendConfig(
            emit_stacktrace=False,
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True,
                skip_d2h_for_method_outputs=True,
            ),
            enable_non_cpu_memory_planning=True,
        )

        for pipeline, program, gm in self._get_executorch_program(
            model, inputs, DeviceAwarePartitioner("cuda:0"), et_config
        ):
            with self.subTest(pipeline=pipeline):
                nodes = _collect_device_copy_nodes(gm)
                self.assertEqual(
                    len(nodes.h2d_nodes),
                    0,
                    f"[{pipeline}] Expected no H2D copy nodes when "
                    f"skip_h2d_for_method_inputs=True, got {len(nodes.h2d_nodes)}",
                )
                self.assertEqual(
                    len(nodes.d2h_nodes),
                    0,
                    f"[{pipeline}] Expected no D2H copy nodes when "
                    f"skip_d2h_for_method_outputs=True, got {len(nodes.d2h_nodes)}",
                )

                # Placeholder inputs should be tagged as CUDA since H2D
                # was skipped.
                cuda_ph, cpu_ph = self._collect_placeholders_by_device(gm)
                self.assertEqual(len(cpu_ph), 0)
                self._assert_nodes_device(
                    cuda_ph,
                    DeviceType.CUDA,
                    pipeline,
                    "Placeholder",
                    expected_index=0,
                )

                # Delegate getitem outputs should stay on CUDA since D2H
                # was skipped.
                getitems = self._collect_delegate_getitems(gm)
                self._assert_nodes_device(
                    getitems,
                    DeviceType.CUDA,
                    pipeline,
                    "Delegate getitem",
                )

                # Verify buffer device mapping: both input and output
                # buffers should be on CUDA.
                self._assert_nodes_buffer_device(
                    cuda_ph,
                    program,
                    DeviceType.CUDA,
                    pipeline,
                    "Placeholder",
                )
                self._assert_nodes_buffer_device(
                    getitems,
                    program,
                    DeviceType.CUDA,
                    pipeline,
                    "Getitem",
                )

    def test_skip_h2d_partial_with_intermediate_input(self):
        """When skip_h2d_for_method_inputs=True, only placeholder inputs
        skip H2D copies. An intermediate (non-placeholder) input feeding
        into the delegate should still get an _h2d_copy node."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                c = torch.sin(a)
                return torch.add(c, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))
        et_config = ExecutorchBackendConfig(
            emit_stacktrace=False,
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True
            ),
            enable_non_cpu_memory_planning=True,
        )

        for pipeline, program, gm in self._get_executorch_program(
            model, inputs, DeviceAwarePartitioner("cuda:0"), et_config
        ):
            with self.subTest(pipeline=pipeline):
                # sin(a) is intermediate (not a placeholder), so it still
                # gets an H2D copy. Placeholder b is skipped.
                nodes = _collect_device_copy_nodes(gm)
                self.assertEqual(
                    len(nodes.h2d_nodes),
                    1,
                    f"[{pipeline}] Expected 1 H2D copy node for the "
                    f"intermediate input, got {len(nodes.h2d_nodes)}",
                )
                self.assertEqual(
                    len(nodes.d2h_nodes),
                    1,
                    f"[{pipeline}] Expected 1 D2H copy node for the single "
                    f"output, got {len(nodes.d2h_nodes)}",
                )

                # Exactly 1 placeholder should be on CUDA (b, which feeds
                # directly into the delegate and skips H2D). The other
                # placeholder (a) feeds through sin() so it stays CPU.
                cuda_ph, cpu_ph = self._collect_placeholders_by_device(gm)
                self.assertEqual(
                    len(cuda_ph),
                    1,
                    f"[{pipeline}] Expected exactly 1 placeholder with CUDA "
                    f"device spec, got {len(cuda_ph)}",
                )

                # Verify buffer device mapping: the CUDA placeholder's
                # buffer should be on CUDA, the CPU placeholder's buffer
                # should be on CPU.
                self._assert_nodes_buffer_device(
                    cuda_ph,
                    program,
                    DeviceType.CUDA,
                    pipeline,
                    "CUDA placeholder",
                )
                self._assert_nodes_buffer_device(
                    cpu_ph,
                    program,
                    DeviceType.CPU,
                    pipeline,
                    "CPU placeholder",
                )

    def test_constant_feeding_device_delegate_is_copied(self):
        """Without skip_h2d_for_constants, a constant feeding a device delegate
        gets an _h2d_copy into planned device memory, with or without
        skip_h2d_for_method_inputs: its data lives in the program on the host,
        so it is not a method input the caller can place."""

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("offset", torch.randn(2, 2))

            def forward(self, a):
                return torch.add(a, self.offset)

        model = Model()
        inputs = (torch.randn(2, 2),)
        for skip in (False, True):
            et_config = ExecutorchBackendConfig(
                emit_stacktrace=False,
                propagate_device_config=PropagateDeviceConfig(
                    skip_h2d_for_method_inputs=skip
                ),
                enable_non_cpu_memory_planning=True,
            )
            with self.assertLogs(propagate_device_pass.logger, "WARNING") as logs:
                results = self._get_executorch_program(
                    model, inputs, DeviceAwarePartitioner("cuda:0"), et_config
                )
            self.assertIn("Constant 'b_offset' (16 bytes)", logs.output[0])
            for pipeline, program, gm in results:
                with self.subTest(pipeline=pipeline, skip=skip):
                    nodes = _collect_device_copy_nodes(gm)
                    # Input a is copied only when the caller cannot place it.
                    self.assertEqual(len(nodes.h2d_nodes), 1 if skip else 2)
                    cuda_ph, _ = self._collect_placeholders_by_device(gm)
                    self.assertEqual(len(cuda_ph), int(skip))
                    self.assertTrue(all(not n.meta["spec"].const for n in cuda_ph))
                    copies = [n.meta["spec"] for n in nodes.h2d_nodes]
                    self.assertTrue(all(not spec.const for spec in copies))
                    self._assert_nodes_buffer_device(
                        nodes.h2d_nodes, program, DeviceType.CUDA, pipeline, "H2D copy"
                    )

    def test_skip_h2d_for_constants_keeps_constant_in_device_memory(self):
        """With skip_h2d_for_constants=True, a buffer or parameter that only
        delegates on one device read, one or several, is planned in that
        device's memory for the whole method, with its data as the initial
        state, so it needs no copy and no warning. Graph inputs being
        unplanned does not change that."""

        class Buffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("offset", torch.randn(2, 2))

            def forward(self, a):
                return torch.add(torch.add(a, self.offset), a)

        class Parameter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.offset = torch.nn.Parameter(torch.randn(2, 2))

            def forward(self, a):
                return torch.add(a, self.offset)

        class TwoDelegates(Buffer):
            def forward(self, a):
                return torch.add(torch.add(a, self.offset), self.offset)

        for alloc_graph_input, model in itertools.product(
            (True, False), (Buffer(), Parameter(), TwoDelegates())
        ):
            with self.assertNoLogs(propagate_device_pass.logger, "WARNING"):
                results = self._get_executorch_program(
                    model,
                    (torch.randn(2, 2),),
                    PerAddDeviceDelegatePartitioner(["cuda:0"]),
                    _skip_h2d_for_constants_config(alloc_graph_input),
                )
            for pipeline, program, gm in results:
                with self.subTest(
                    pipeline=pipeline,
                    alloc_input=alloc_graph_input,
                    model=type(model).__name__,
                ):
                    self.assertEqual(_collect_device_copy_nodes(gm).h2d_nodes, [])
                    (offset,) = [
                        n
                        for n in gm.graph.nodes
                        if n.op == "placeholder" and n.name != "a"
                    ]
                    spec = offset.meta["spec"]
                    self.assertTrue(spec.const)
                    self.assertEqual(spec.lifetime, [0, len(gm.graph.nodes) - 1])
                    self._assert_buffer_device(
                        spec, program, DeviceType.CUDA, "device resident constant"
                    )
                    (tensor,) = [
                        value.val
                        for value in program.execution_plan[0].values
                        if getattr(value.val, "allocation_info", None) is not None
                        and value.val.data_buffer_idx > 0
                    ]
                    self.assertEqual(
                        tensor.extra_tensor_info.device_type, DeviceType.CUDA
                    )

    def test_skip_h2d_for_constants_copies_constants_read_elsewhere(self):
        """A constant that a host op, or a delegate on another device, reads
        too is still copied with skip_h2d_for_constants=True, and so is every
        constant without the option. The warning says why the option did not
        keep it, or suggests the option when it is off."""

        class Buffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("offset", torch.randn(2, 2))

            def forward(self, a):
                return torch.add(a, self.offset)

        class AlsoReadOnHost(Buffer):
            def forward(self, a):
                return torch.add(a, self.offset), self.offset * 2

        class ReadByTwoDelegates(Buffer):
            def forward(self, a):
                return torch.add(a, self.offset), torch.add(a * 2, self.offset)

        class EmptyBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("offset", torch.zeros(2, 0))

            def forward(self, a):
                return torch.add(a[:, :0], self.offset)

        kept_elsewhere = "skip_h2d_for_constants cannot keep it on the device"
        cases = {
            "host reader": (AlsoReadOnHost(), ["cuda:0"], True, [0], kept_elsewhere),
            "two devices": (
                ReadByTwoDelegates(),
                ["cuda:0", "cuda:1"],
                True,
                [0, 1],
                kept_elsewhere,
            ),
            "without the option": (
                Buffer(),
                ["cuda:0"],
                False,
                [0],
                "or set PropagateDeviceConfig(skip_h2d_for_constants=True)",
            ),
            # The option would not help, so it is not suggested.
            "host reader without the option": (
                AlsoReadOnHost(),
                ["cuda:0"],
                False,
                [0],
                "take it into the delegate to avoid that copy",
            ),
            # Nor for an empty constant, which costs nothing to copy.
            "empty without the option": (
                EmptyBuffer(),
                ["cuda:0"],
                False,
                [0],
                "take it into the delegate to avoid that copy",
            ),
        }
        for case, (model, devices, skip_constants, copied_to, hint) in cases.items():
            with self.assertLogs(propagate_device_pass.logger, "WARNING") as logs:
                results = self._get_executorch_program(
                    model,
                    (torch.randn(2, 2),),
                    PerAddDeviceDelegatePartitioner(devices),
                    _skip_h2d_for_constants_config(False, skip_constants),
                )
            warnings = [w for w in logs.output if "Constant 'b_offset'" in w]
            self.assertTrue(warnings)
            for warning in warnings:
                self.assertIn(hint, warning)
                self.assertNotIn("_h2d_copy", warning)
            for pipeline, _program, gm in results:
                with self.subTest(pipeline=pipeline, case=case):
                    copies = [
                        n.meta["spec"].device_index
                        for n in _collect_device_copy_nodes(gm).h2d_nodes
                        if n.args[0].name == "b_offset"
                    ]
                    self.assertEqual(sorted(copies), copied_to)

    def test_skip_h2d_for_constants_on_its_own(self):
        """skip_h2d_for_constants works without skip_h2d_for_method_inputs:
        the method input is still copied to the device and the constant is
        not. A constant that is still copied raises with
        error_on_per_call_copies."""

        class Buffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("offset", torch.randn(2, 2))

            def forward(self, a):
                return torch.add(a, self.offset)

        class AlsoReadOnHost(Buffer):
            def forward(self, a):
                return torch.add(a, self.offset), self.offset * 2

        def config(error_on_per_call_copies):
            return ExecutorchBackendConfig(
                emit_stacktrace=False,
                propagate_device_config=PropagateDeviceConfig(
                    skip_h2d_for_constants=True,
                    error_on_per_call_copies=error_on_per_call_copies,
                ),
                enable_non_cpu_memory_planning=True,
            )

        inputs = (torch.randn(2, 2),)
        for pipeline, _program, gm in self._get_executorch_program(
            Buffer(), inputs, DeviceAwarePartitioner("cuda:0"), config(False)
        ):
            with self.subTest(pipeline=pipeline):
                (copy,) = _collect_device_copy_nodes(gm).h2d_nodes
                self.assertEqual(copy.args[0].name, "a")
        with self.assertRaisesRegex(
            RuntimeError, "Constant 'b_offset'.*error_on_per_call_copies=True"
        ):
            self._get_executorch_program(
                AlsoReadOnHost(), inputs, DeviceAwarePartitioner("cuda:0"), config(True)
            )

    def test_error_on_per_call_copies(self):
        """error_on_per_call_copies turns the warning about a copy that a skip
        flag cannot remove into an error: a constant copied to the device and
        an input copied for a host reader under skip_h2d_for_method_inputs, and
        a returned output written on the host under skip_d2h_for_method_outputs.
        Without the skip flag, a constant copy only logs the warning."""

        class ReadsConstant(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("offset", torch.randn(2, 2))

            def forward(self, a):
                return torch.add(a, self.offset), self.offset * 2

        class WritesOutputOnHost(torch.nn.Module):
            def forward(self, x, y):
                a = x + y
                torch.ops.propagate_device_test.accumulate(a, y)
                return a

        class ReadsInputOnHost(torch.nn.Module):
            def forward(self, x, y):
                return x + y, torch.sin(x)

        def lower(model, num_inputs, **flags):
            return _lower_with_host_writer(
                model,
                tuple(torch.randn(2, 2) for _ in range(num_inputs)),
                ["cuda:0"],
                _host_writer_config(
                    False,
                    propagate_device_config=PropagateDeviceConfig(
                        error_on_per_call_copies=True, **flags
                    ),
                ),
            )

        cases = {
            "constant": (ReadsConstant(), 1, "skip_h2d_for_method_inputs"),
            "input read on host": (ReadsInputOnHost(), 2, "skip_h2d_for_method_inputs"),
            "output written on host": (
                WritesOutputOnHost(),
                2,
                "skip_d2h_for_method_outputs",
            ),
        }
        for case, (model, num_inputs, flag) in cases.items():
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "on every call.*Raised because error_on_per_call_copies=True",
                ):
                    lower(model, num_inputs, **{flag: True})
        with self.assertLogs(propagate_device_pass.logger, "WARNING"):
            lower(ReadsConstant(), 1)

    def test_constant_copy_keeps_stored_layout(self):
        """The copy of a constant moves its bytes as they are, so it must use the
        layout the constant is stored in: contiguous for a transposed parameter,
        which to_edge makes contiguous, and channels-last for a channels-last
        one."""

        class Model(torch.nn.Module):
            def __init__(self, weight):
                super().__init__()
                self.weight = torch.nn.Parameter(weight)

            def forward(self, a):
                return torch.add(a, self.weight)

        weights = {
            "transposed": torch.randn(3, 2).t(),
            "channels_last": torch.randn(1, 2, 3, 3).to(
                memory_format=torch.channels_last
            ),
        }
        for name, weight in weights.items():
            et_config = ExecutorchBackendConfig(
                emit_stacktrace=False,
                propagate_device_config=PropagateDeviceConfig(
                    skip_h2d_for_method_inputs=True
                ),
                enable_non_cpu_memory_planning=True,
            )
            for pipeline, _program, gm in self._get_executorch_program(
                Model(weight),
                (torch.randn(weight.shape),),
                DeviceAwarePartitioner("cuda:0"),
                et_config,
            ):
                with self.subTest(pipeline=pipeline, weight=name):
                    nodes = _collect_device_copy_nodes(gm)
                    self.assertEqual(len(nodes.h2d_nodes), 1)
                    h2d = nodes.h2d_nodes[0]
                    # The emitter rewrites the constant's spec to the stored layout.
                    stored = h2d.args[0].meta["spec"]
                    self.assertEqual(h2d.meta["spec"].dim_order, stored.dim_order)
                    self.assertEqual(
                        tuple(h2d.meta["spec"].stride), tuple(stored.stride)
                    )

    def test_constant_with_unstored_layout_is_refused(self):
        """A constant that is neither contiguous nor channels-last is stored
        contiguous, so its copy would not match the stored data. A pass that runs
        after to_edge normalized the constants can create one, and lowering then
        fails instead of producing wrong results."""

        class StoreWeightTransposed(ExportedProgramPassBase):
            def call(self, ep):
                ep.state_dict["w"] = torch.nn.Parameter(
                    ep.state_dict["w"].t().contiguous().t()
                )
                for node in ep.graph.nodes:
                    if node.name == "p_w":
                        node.meta["val"] = node.meta["val"].t().contiguous().t()
                return ExportedProgramPassResult(ep, True)

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.randn(2, 3))

            def forward(self, a):
                return torch.add(a, self.w)

        for skip in (False, True):
            edge = to_edge_transform_and_lower(
                export(Model(), (torch.randn(2, 3),)),
                transform_passes=[StoreWeightTransposed()],
                partitioner=[DeviceAwarePartitioner("cuda:0")],
            )
            et_config = ExecutorchBackendConfig(
                emit_stacktrace=False,
                propagate_device_config=PropagateDeviceConfig(
                    skip_h2d_for_method_inputs=skip
                ),
                enable_non_cpu_memory_planning=True,
            )
            with self.subTest(skip=skip):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Constant 'p_w' feeds a delegate on cuda:0, but it is neither "
                    "contiguous nor channels-last",
                ):
                    edge.to_executorch(et_config)

    def test_constant_read_by_several_delegates_is_copied_once_per_device(self):
        """A constant read by several delegates gets one _h2d_copy for each
        device, not one for each delegate."""

        inputs = (torch.randn(2, 2), torch.randn(2, 2))
        lowered = {
            device: _lower_add_to_device(inputs, device)
            for device in ("cuda:0", "cuda:1")
        }

        class Model(torch.nn.Module):
            def __init__(self, devices):
                super().__init__()
                self.register_buffer("offset", torch.randn(2, 2))
                self.first, self.second, self.third = (lowered[d] for d in devices)

            def forward(self, x):
                return (
                    self.first(x, self.offset),
                    self.second(x, self.offset),
                    self.third(x, self.offset),
                )

        et_config = ExecutorchBackendConfig(
            emit_stacktrace=False, enable_non_cpu_memory_planning=True
        )
        cases = {
            "one device": (("cuda:0", "cuda:0", "cuda:0"), [0]),
            "two devices": (("cuda:0", "cuda:1", "cuda:0"), [0, 1]),
        }
        for case, (devices, copy_indices) in cases.items():
            with self.assertLogs(propagate_device_pass.logger, "WARNING") as logs:
                et = to_edge(export(Model(devices), inputs[:1])).to_executorch(
                    deepcopy(et_config)
                )
            gm = et.exported_program().graph_module
            with self.subTest(case=case):
                self.assertEqual(len(logs.output), len(copy_indices))
                nodes = _collect_device_copy_nodes(gm)
                offset_copies = [
                    n for n in nodes.h2d_nodes if n.args[0].name == "b_offset"
                ]
                self.assertEqual(
                    sorted(n.meta["spec"].device_index for n in offset_copies),
                    copy_indices,
                )
                for delegate, device in zip(nodes.delegate_nodes, devices):
                    offset = delegate.args[2]
                    self.assertIn(offset, offset_copies)
                    self.assertEqual(f"cuda:{offset.meta['spec'].device_index}", device)

    def test_tensor_written_on_host_is_copied_for_each_delegate(self):
        """A tensor that a host op writes in place, directly or through a view,
        between two delegates on one device gets a new copy for the second
        delegate, so it reads the new value."""

        class Model(torch.nn.Module):
            def __init__(self, write_view):
                super().__init__()
                self.write_view = write_view

            def forward(self, x, y):
                t = x * 2
                a = t + y
                if self.write_view:
                    torch.ops.propagate_device_test.accumulate(t.view(-1), a.view(-1))
                else:
                    torch.ops.propagate_device_test.accumulate(t, a)
                return a, t + y

        for write_view in (False, True):
            gm = _lower_with_host_writer(
                Model(write_view),
                (torch.randn(2, 2), torch.randn(2, 2)),
                ["cuda:0", "cuda:0"],
                _host_writer_config(write_view),
            )
            with self.subTest(write_view=write_view):
                writer = _host_writer(gm)
                first, second = _collect_device_copy_nodes(gm).delegate_nodes
                (second_copy,) = [
                    arg
                    for arg in second.args[1:]
                    if arg.target == torch.ops.et_copy._h2d_copy.out
                    and arg.args[0] is _host_written(gm)
                ]
                nodes = list(gm.graph.nodes)
                self.assertLess(nodes.index(first), nodes.index(writer))
                self.assertLess(nodes.index(writer), nodes.index(second_copy))

    def test_skip_h2d_for_input_shared_by_device_delegates(self):
        """With skip_h2d_for_method_inputs=True, a method input read by two
        delegates on the same device is placed on that device once, with no
        _h2d_copy."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                c = torch.sin(torch.add(a, b))
                return torch.add(a, c)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))
        et_config = ExecutorchBackendConfig(
            emit_stacktrace=False,
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True
            ),
            enable_non_cpu_memory_planning=True,
        )
        for pipeline, program, gm in self._get_executorch_program(
            model, inputs, DeviceAwarePartitioner("cuda:0"), et_config
        ):
            with self.subTest(pipeline=pipeline):
                nodes = _collect_device_copy_nodes(gm)
                self.assertEqual(len(nodes.delegate_nodes), 2)
                # Only sin's output, computed on the host, is copied.
                self.assertEqual(len(nodes.h2d_nodes), 1)
                cuda_ph, _ = self._collect_placeholders_by_device(gm)
                self.assertEqual(sorted(n.name for n in cuda_ph), ["a", "b"])
                self._assert_nodes_buffer_device(
                    cuda_ph, program, DeviceType.CUDA, pipeline, "Placeholder"
                )

    def test_skip_h2d_copies_input_for_readers_off_device(self):
        """With skip_h2d_for_method_inputs=True, a method input that a device
        delegate reads stays on that device. A host op, a delegate on the host,
        or a delegate on another device that also reads it reads one _d2h_copy
        of it, also when more than one delegate on the device reads it, and a
        warning says so."""

        class HostOp(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b), torch.sin(a)

        class TwoDelegates(torch.nn.Module):
            def forward(self, a, b, c):
                return torch.add(a, b), torch.add(a, c)

        class TwoDelegatesAndHostOp(torch.nn.Module):
            def forward(self, a, b, c):
                return torch.add(a, b), torch.add(a, c), torch.sin(a)

        class CopiedIntoBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(2, 2))

            def forward(self, a, b):
                self.state.copy_(a)
                return torch.add(a, b)

        et_config = ExecutorchBackendConfig(
            emit_stacktrace=False,
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True
            ),
            enable_non_cpu_memory_planning=True,
        )
        cases = {
            "host op": (HostOp(), 2, DeviceAwarePartitioner("cuda:0")),
            "two delegates on the device and a host op": (
                TwoDelegatesAndHostOp(),
                3,
                PerAddDeviceDelegatePartitioner(["cuda:0", "cuda:0"]),
            ),
            "delegate without a device": (
                TwoDelegates(),
                3,
                PerAddDeviceDelegatePartitioner(["cuda:0", None]),
            ),
            "delegate on the cpu": (
                TwoDelegates(),
                3,
                PerAddDeviceDelegatePartitioner(["cuda:0", "cpu"]),
            ),
            "other device index": (
                TwoDelegates(),
                3,
                PerAddDeviceDelegatePartitioner(["cuda:0", "cuda:1"]),
            ),
            # The write-back of the buffer only reads the input.
            "buffer write-back": (
                CopiedIntoBuffer(),
                2,
                DeviceAwarePartitioner("cuda:0"),
            ),
        }
        for case, (model, num_inputs, partitioner) in cases.items():
            inputs = tuple(torch.randn(2, 2) for _ in range(num_inputs))
            with self.assertLogs(propagate_device_pass.logger, "WARNING") as logs:
                results = self._get_executorch_program(
                    model, inputs, partitioner, et_config
                )
            self.assertIn("Method input 'a' (16 bytes) stays on cuda:0", logs.output[0])
            for pipeline, program, gm in results:
                with self.subTest(case=case, pipeline=pipeline):
                    a = next(n for n in gm.graph.nodes if n.name == "a")
                    self.assertEqual(a.meta["spec"].device, DeviceType.CUDA)
                    self.assertEqual(a.meta["spec"].device_index, 0)
                    nodes = _collect_device_copy_nodes(gm)
                    self.assertIs(nodes.delegate_nodes[0].args[1], a)
                    # The copy is the only other reader of the input, and is
                    # not copied again.
                    (copy,) = [n for n in a.users if n not in nodes.delegate_nodes]
                    self.assertIn(copy, nodes.d2h_nodes)
                    self.assertFalse(set(copy.users) & set(nodes.d2h_nodes))
                    self._assert_nodes_buffer_device(
                        [copy], program, DeviceType.CPU, pipeline, "D2H copy"
                    )

    def test_skip_h2d_shape_reader_needs_no_copy(self):
        """With skip_h2d_for_method_inputs=True, an op that reads only the shape
        of an input on the device reads the input there, since its sizes stay
        in host memory, so the input gets no copy and no warning."""

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.add(x, y), torch.zeros(x.shape[0])

        et_config = ExecutorchBackendConfig(
            emit_stacktrace=False,
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True
            ),
            enable_non_cpu_memory_planning=True,
        )
        batch = torch.export.Dim("batch", min=2, max=8)
        ep = export(
            Model(),
            (torch.randn(4, 2), torch.randn(4, 2)),
            dynamic_shapes={"x": {0: batch}, "y": {0: batch}},
        )
        edge = to_edge(
            ep, compile_config=EdgeCompileConfig(_check_ir_validity=False)
        ).to_backend(DeviceAwarePartitioner("cuda:0"))
        with self.assertNoLogs(propagate_device_pass.logger, "WARNING"):
            gm = edge.to_executorch(et_config).exported_program().graph_module
        (size,) = [n for n in gm.graph.nodes if n.target == torch.ops.aten.sym_size.int]
        read = size.args[0]
        self.assertEqual(read.op, "placeholder")
        self.assertEqual(read.meta["spec"].device, DeviceType.CUDA)
        self.assertFalse(
            set(read.users) & set(_collect_device_copy_nodes(gm).d2h_nodes)
        )

    def test_skip_both_returns_input_on_device_without_copy(self):
        """With both skip flags, a method that returns an input a device
        delegate also reads returns it on the device, with no copy and no
        warning."""

        class Model(torch.nn.Module):
            def forward(self, x, y):
                return torch.add(x, y), x

        et_config = ExecutorchBackendConfig(
            emit_stacktrace=False,
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True, skip_d2h_for_method_outputs=True
            ),
            enable_non_cpu_memory_planning=True,
        )
        with self.assertNoLogs(propagate_device_pass.logger, "WARNING"):
            results = self._get_executorch_program(
                Model(),
                (torch.randn(2, 2), torch.randn(2, 2)),
                DeviceAwarePartitioner("cuda:0"),
                et_config,
            )
        for pipeline, _program, gm in results:
            with self.subTest(pipeline=pipeline):
                nodes = _collect_device_copy_nodes(gm)
                self.assertEqual(nodes.h2d_nodes + nodes.d2h_nodes, [])
                x = next(n for n in gm.graph.nodes if n.name == "x")
                self.assertIn(x, gm.graph.output_node().args[0])
                self.assertEqual(x.meta["spec"].device, DeviceType.CUDA)

    def test_skip_h2d_refuses_placeholder_written_on_host(self):
        """A mutated input or mutable buffer is written back on the host, so the
        refusal says so, also when a host op reads it too. An input that a host
        op writes in place is refused as well, since the write would only reach
        its host copy."""

        class MutableBuffer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(2, 2))

            def forward(self, a):
                out = torch.add(a, self.state)
                self.state.add_(a)
                return out

        class MutableBufferReadOnHost(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("state", torch.zeros(2, 2))

            def forward(self, a):
                out = torch.add(a, self.state)
                read = torch.sin(self.state)
                self.state.add_(a)
                return out, read

        class MutatedInput(torch.nn.Module):
            def forward(self, a, x):
                out = torch.add(a, x)
                x.add_(a)
                return out

        class InputWrittenByHostOp(torch.nn.Module):
            def forward(self, a, x):
                out = torch.add(a, x)
                torch.ops.propagate_device_test.accumulate(x, a)
                return out

        et_config = ExecutorchBackendConfig(
            emit_stacktrace=False,
            propagate_device_config=PropagateDeviceConfig(
                skip_h2d_for_method_inputs=True
            ),
            enable_non_cpu_memory_planning=True,
        )
        write_back = r"'copy_' writes it back on the host \(the method mutates it\)"
        cases = {
            "mutable buffer": (MutableBuffer(), 1, "b_state", write_back),
            "mutable buffer read on host": (
                MutableBufferReadOnHost(),
                1,
                "b_state",
                write_back,
            ),
            "mutated input": (MutatedInput(), 2, "x", write_back),
            "input written by a host op": (
                InputWrittenByHostOp(),
                2,
                "x",
                "'[a-z_]*accumulate[a-z_]*' writes it in place on the host",
            ),
        }
        for case, (model, num_inputs, name, reason) in cases.items():
            inputs = tuple(torch.randn(2, 2) for _ in range(num_inputs))
            edge = to_edge(
                export(model, inputs),
                compile_config=EdgeCompileConfig(_check_ir_validity=False),
            ).to_backend(DeviceAwarePartitioner("cuda:0"))
            with self.subTest(case=case):
                with self.assertRaisesRegex(
                    RuntimeError, f"placeholder '{name}' on cuda:0, because {reason}"
                ):
                    edge.to_executorch(deepcopy(et_config))

    def test_tensorspec_repr_includes_device(self):
        spec = TensorSpec(dtype=torch.float32, shape=torch.Size([2, 3]))
        repr_str = repr(spec)
        self.assertIn("device=", repr_str)
        self.assertIn("CPU", repr_str)

    def test_skipping_copies_requires_unallocated_graph_io(self):
        """Skipping a boundary copy is only complete when the runtime also stops reserving its own
        buffer for that tensor.

        Memory planning allocates graph inputs and outputs by default. A planned input carries
        allocation info, and the runtime fills a planned input by copying the caller's memory into the
        buffer it reserved, so the copy the export asked to skip comes back at run time. For device
        memory that copy is also a host memcpy into a device pointer, which crashes.

        This pins the pairing the configuration needs, because the two settings live in different
        places and nothing else connects them.
        """

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        model = Model()
        inputs = (torch.randn(2, 2), torch.randn(2, 2))
        skip_copies = PropagateDeviceConfig(
            skip_h2d_for_method_inputs=True,
            skip_d2h_for_method_outputs=True,
        )

        def planned_io(config: ExecutorchBackendConfig):
            """Whether the program reserves its own buffer for each graph input and output."""
            lowered = to_edge_transform_and_lower(
                torch.export.export(model, inputs),
                partitioner=[DeviceAwarePartitioner("cuda:0")],
            ).to_executorch(config)
            plan = lowered.executorch_program.execution_plan[0]
            planned = lambda indices: [  # noqa: E731
                getattr(plan.values[index].val, "allocation_info", None) is not None
                for index in indices
            ]
            return planned(plan.inputs), planned(plan.outputs)

        # Default planning reserves buffers, so the runtime copies into them despite the skip flags.
        planned_inputs, planned_outputs = planned_io(
            ExecutorchBackendConfig(
                propagate_device_config=skip_copies,
                enable_non_cpu_memory_planning=True,
            )
        )
        self.assertTrue(
            all(planned_inputs),
            "graph inputs are expected to be planned by default, which is why the pairing below "
            f"is needed, but got {planned_inputs}",
        )

        # Asking planning to leave them alone is what actually removes the copy.
        planned_inputs, planned_outputs = planned_io(
            ExecutorchBackendConfig(
                propagate_device_config=skip_copies,
                enable_non_cpu_memory_planning=True,
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=False
                ),
            )
        )
        self.assertFalse(
            any(planned_inputs),
            f"no graph input should be planned when the caller provides it, got {planned_inputs}",
        )
        self.assertFalse(
            any(planned_outputs),
            f"no graph output should be planned when it stays on the device, got {planned_outputs}",
        )

    def test_device_output_copied_to_host_is_planned(self):
        """A delegate output copied to the host is an intermediate tensor on the
        device, so memory planning must place it even when the graph outputs are
        left to the caller. Unplanned, the runtime hands the delegate a null
        pointer for it."""

        class Model(torch.nn.Module):
            def forward(self, a, b):
                return torch.add(a, b)

        lowered = to_edge_transform_and_lower(
            torch.export.export(Model(), (torch.randn(2, 2), torch.randn(2, 2))),
            partitioner=[DeviceAwarePartitioner("cuda:0")],
        ).to_executorch(
            ExecutorchBackendConfig(
                propagate_device_config=PropagateDeviceConfig(
                    skip_h2d_for_method_inputs=True
                ),
                enable_non_cpu_memory_planning=True,
                memory_planning_pass=MemoryPlanningPass(
                    alloc_graph_input=False, alloc_graph_output=False
                ),
            )
        )
        plan = lowered.executorch_program.execution_plan[0]
        device_intermediates = [
            index
            for index, value in enumerate(plan.values)
            if index not in plan.inputs
            and getattr(value.val, "extra_tensor_info", None) is not None
            and value.val.extra_tensor_info.device_type == DeviceType.CUDA
        ]
        self.assertTrue(device_intermediates)
        for index in device_intermediates:
            self.assertIsNotNone(
                plan.values[index].val.allocation_info,
                f"device tensor {index} is not planned",
            )
        for index in plan.outputs:
            self.assertIsNone(plan.values[index].val.allocation_info)


if __name__ == "__main__":
    unittest.main()
