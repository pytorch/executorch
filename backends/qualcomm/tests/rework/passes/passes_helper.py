# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from typing import OrderedDict

import torch
from executorch.backends.qualcomm._passes.qnn_pass_manager import (
    get_qnn_pass_manager_cls,
)
from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner
from executorch.backends.qualcomm.serialization.qc_schema import (
    QnnExecuTorchBackendType,
)
from executorch.backends.qualcomm.tests.rework.conftest import calibrate
from executorch.backends.qualcomm.utils.qnn_manager_lifecycle import QnnManagerContext
from executorch.backends.qualcomm.utils.utils import qnn_edge_config
from executorch.exir.backend.backend_details import CompileSpec
from executorch.exir.pass_base import ExportPass
from executorch.exir.program._program import _gen_edge_manager_for_partitioners
from torchao.quantization.pt2e.quantizer import Quantizer

NodeTarget = object | set[object]


class Assertions:
    @staticmethod
    def count_nodes(gm: torch.fx.GraphModule, target: NodeTarget) -> int:
        targets = target if isinstance(target, set) else {target}
        return sum(
            1 for n in gm.graph.nodes if n.op == "call_function" and n.target in targets
        )

    @staticmethod
    def assert_no_target(gm: torch.fx.GraphModule, target: NodeTarget) -> None:
        n = Assertions.count_nodes(gm, target)
        assert n == 0, f"expected no {target} nodes, found {n}"

    @staticmethod
    def assert_target_count(
        gm: torch.fx.GraphModule, target: NodeTarget, expect_count: int
    ) -> None:
        target_found = Assertions.count_nodes(gm, target)
        assert (
            expect_count == target_found
        ), f"{target} expecting count: {expect_count}, but found {target_found}."

    @staticmethod
    def assert_target_count_at_most(
        gm: torch.fx.GraphModule, target: NodeTarget, max_count: int
    ) -> None:
        """Upper bound on node count; use when the exact count is fragile but a
        ceiling is the real contract (e.g. a fuse pass reducing N nodes)."""
        target_found = Assertions.count_nodes(gm, target)
        assert (
            target_found <= max_count
        ), f"{target} expecting at most {max_count}, but found {target_found}."

    @staticmethod
    def assert_target_count_at_least(
        gm: torch.fx.GraphModule, target: NodeTarget, min_count: int
    ) -> None:
        """Lower bound on node count; use when the exact count is fragile but a
        floor is the real contract (e.g. a pass that inserts at least N QDQ
        pairs around fallback nodes)."""
        target_found = Assertions.count_nodes(gm, target)
        assert (
            target_found >= min_count
        ), f"{target} expecting at least {min_count}, but found {target_found}."

    @staticmethod
    def assert_no_consecutive(gm: torch.fx.GraphModule, target: NodeTarget) -> None:
        """Assert no node in `target` feeds directly into another node in
        `target`. This is the structural invariant a fuse-consecutive pass
        enforces; a plain count can pass while the structure is still wrong."""
        targets = target if isinstance(target, set) else {target}
        for n in gm.graph.nodes:
            if n.op != "call_function" or n.target not in targets:
                continue
            for user in n.users:
                assert not (
                    user.op == "call_function" and user.target in targets
                ), f"found consecutive {target}: {n} -> {user}"

    @staticmethod
    def assert_target_dtype(
        gm: torch.fx.GraphModule, target: NodeTarget, dtype: torch.dtype
    ) -> None:
        """Assert every node in `target` outputs a tensor of `dtype`. Catches
        dtype-canonicalization invariants (e.g. i64 casts rewritten to i32)."""
        targets = target if isinstance(target, set) else {target}
        for n in gm.graph.nodes:
            if n.op != "call_function" or n.target not in targets:
                continue
            val = n.meta.get("val")
            got = getattr(val, "dtype", None)
            assert got == dtype, f"{target} node {n} expected dtype {dtype}, got {got}"

    @staticmethod
    def assert_meta_key_on_targets(
        gm: torch.fx.GraphModule, target: NodeTarget, meta_key: str
    ) -> None:
        """Assert every node whose target is in `target` has `meta_key` in
        node.meta. Fails if no matching target node exists (would trivially
        pass otherwise). Used to verify tagging passes like LayoutTransform
        set the expected meta key on layout-sensitive ops."""
        targets = target if isinstance(target, set) else {target}
        found = 0
        for n in gm.graph.nodes:
            if n.op != "call_function" or n.target not in targets:
                continue
            found += 1
            assert (
                meta_key in n.meta
            ), f"{target} node {n} missing meta key {meta_key!r}"
        assert found > 0, f"no nodes with target {target} found to check meta"

    @staticmethod
    def assert_meta_key_on_named_node(
        gm: torch.fx.GraphModule, node_name: str, meta_key: str
    ) -> None:
        """Assert the node with `node_name` exists and has `meta_key` in
        node.meta. More surgical than assert_meta_key_on_targets: identifies
        the exact node by its FX name rather than by op target.

        Useful for verifying tags on nodes selected by identity (e.g. a
        skip_node_id_set entry) rather than by op kind."""
        for n in gm.graph.nodes:
            if n.name == node_name:
                assert (
                    meta_key in n.meta
                ), f"node {node_name!r} missing meta key {meta_key!r}"
                return
        raise AssertionError(f"node {node_name!r} not found in graph")


class PassPipeline:
    @staticmethod
    def _instantiate(pass_classes, **available_kwargs):
        """Instantiate pass classes, injecting only kwargs each __init__ accepts."""
        instances = []
        for p_cls in pass_classes:
            init_params = inspect.signature(p_cls.__init__).parameters
            kwargs = {k: v for k, v in available_kwargs.items() if k in init_params}
            instances.append(p_cls(**kwargs))
        return instances

    @staticmethod
    def _slice_to_target(items, target_pass, pipeline_name, match_fn=None):
        """Trim `items` to include only up to (and including) the first item
        matching `target_pass`. Raises AssertionError if no match is found.

        Default match compares class identity (`is`); pass a predicate via
        `match_fn` for other cases (e.g. `isinstance` on constructed instances).
        """
        if match_fn is None:
            match_fn = lambda item: item is target_pass  # noqa: E731
        idx = next(
            (i for i, item in enumerate(items) if match_fn(item)),
            None,
        )
        assert (
            idx is not None
        ), f"{target_pass.__name__} not found in {pipeline_name} passes"
        return items[: idx + 1]

    @staticmethod
    def lower_annotation_gm(
        module: torch.nn.Module,
        sample_input: tuple[torch.Tensor, ...],
        target_pass: type[ExportPass],
        backend_type: QnnExecuTorchBackendType = QnnExecuTorchBackendType.kHtpBackend,
    ) -> torch.fx.GraphModule:
        pm_cls = get_qnn_pass_manager_cls(backend_type)
        gm = torch.export.export(module, sample_input, strict=True).module()
        pass_classes = PassPipeline._slice_to_target(
            pm_cls.get_annotation_passes(), target_pass, "annotation"
        )
        instances = PassPipeline._instantiate(
            pass_classes,
            quantization_capture=True,
        )
        for p in instances:
            gm = p(gm).graph_module
        return gm

    @staticmethod
    def lower_export_gm(
        module: torch.nn.Module,
        sample_input: tuple[torch.Tensor, ...],
        target_pass: type[ExportPass],
        backend_type: QnnExecuTorchBackendType = QnnExecuTorchBackendType.kHtpBackend,
        quantizer: Quantizer = None,
        convert_linear_to_conv2d: bool = False,
    ) -> torch.fx.GraphModule:
        with calibrate(module, [sample_input], quantizer) as quantized:
            module = quantized
        ep = torch.export.export(module, sample_input, dynamic_shapes=None, strict=True)
        gm = ep.graph_module
        pm_cls = get_qnn_pass_manager_cls(backend_type)
        pass_classes = PassPipeline._slice_to_target(
            pm_cls.get_export_passes(convert_linear_to_conv2d=convert_linear_to_conv2d),
            target_pass,
            "export",
        )
        instances = PassPipeline._instantiate(
            pass_classes,
            edge_program=ep,
            quantization_capture=True,
        )
        for p in instances:
            gm = p(gm).graph_module
        return gm

    @staticmethod
    def lower_edge_ep(
        module: torch.nn.Module,
        sample_input: tuple[torch.Tensor, ...],
        backend_type: QnnExecuTorchBackendType,
        compile_spec: CompileSpec,
        target_pass: type[ExportPass] | None = None,
        quantizer: Quantizer = None,
        passes_job: OrderedDict = None,
        skip_node_id_set: set | None = None,
        skip_node_op_set: set | None = None,
    ):
        """
        If target_pass is None, runs all edge passes.
        None should only be used by PassPipeline.lower_preprocess_gm, not individual tests.
        """
        with calibrate(module, [sample_input], quantizer) as quantized:
            module = quantized
        ep = torch.export.export(module, sample_input, dynamic_shapes=None, strict=True)
        pm = get_qnn_pass_manager_cls(backend_type)()
        ep = pm.transform_for_export_pipeline(exported_program=ep)
        edge_passes = pm.get_to_edge_transform_passes(
            ep,
            compiler_specs=compile_spec,
            passes_job=passes_job,
            skip_node_id_set=skip_node_id_set,
            skip_node_op_set=skip_node_op_set,
        )
        if target_pass is not None:
            edge_passes = PassPipeline._slice_to_target(
                edge_passes,
                target_pass,
                "edge",
                match_fn=lambda p: isinstance(p, target_pass),
            )
            assert (
                edge_passes is not None
            ), f"{target_pass.__name__} not found in edge passes"
        qnn_partitioner = QnnPartitioner(
            compile_spec,
            skip_node_id_set=skip_node_id_set,
            skip_node_op_set=skip_node_op_set,
            skip_mutable_buffer=False,
        )
        forward_method = "forward"
        with QnnManagerContext({forward_method: compile_spec}):
            config = qnn_edge_config()
            edge_manager = _gen_edge_manager_for_partitioners(
                partitioner={forward_method: [qnn_partitioner]},
                aten_programs={forward_method: ep},
                config=config,
                constant_methods=None,
            )
            edge_manager = edge_manager.transform(edge_passes)
        return edge_manager.exported_program()

    @staticmethod
    def lower_preprocess_gm(
        module: torch.nn.Module,
        sample_input: tuple[torch.Tensor, ...],
        backend_type: QnnExecuTorchBackendType,
        compile_spec: CompileSpec,
        target_pass: type[ExportPass],
        quantizer: Quantizer = None,
        use_mha2sha: bool = False,
    ) -> torch.fx.GraphModule:
        edge_ep = PassPipeline.lower_edge_ep(
            module=module,
            sample_input=sample_input,
            target_pass=None,
            backend_type=backend_type,
            compile_spec=compile_spec,
            quantizer=quantizer,
        )
        gm = edge_ep.graph_module
        # Mirror qnn_preprocess.py: strip QCOM_AXIS_ORDER from all nodes
        from executorch.backends.qualcomm.utils.constants import QCOM_AXIS_ORDER

        for node in gm.graph.nodes:
            if hasattr(node, "meta"):
                node.meta.pop(QCOM_AXIS_ORDER, "")

        pm_cls = get_qnn_pass_manager_cls(backend_type)
        pass_classes = PassPipeline._slice_to_target(
            pm_cls.get_preprocess_passes(use_mha2sha=use_mha2sha),
            target_pass,
            "preprocess",
        )
        instances = PassPipeline._instantiate(
            pass_classes,
            edge_program=edge_ep,
            force_fold=True,
            insert_permute=True,
        )
        for p in instances:
            gm = p(gm).graph_module
        return gm
