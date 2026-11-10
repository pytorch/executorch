# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright 2024-2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile
from collections import Counter
from typing import Any, Callable, Dict, get_args, List, Optional, Sequence, Tuple, Type

import executorch
import executorch.backends.test.harness.stages as BaseStages

import torch
from executorch.backends.test.harness import Tester as TesterBase
from executorch.backends.test.harness.stages import StageType
from executorch.backends.xnnpack._passes import XNNPACKPassManager
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
    get_symmetric_quantization_config,
    XNNPACKQuantizer,
)
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer_utils import (
    QuantizationConfig,
)
from executorch.backends.xnnpack.serialization.xnnpack_graph_schema import XNodeUnion
from executorch.backends.xnnpack.serialization.xnnpack_graph_serialize import (
    _get_schema_bytes,
    XNNHeader,
)
from executorch.backends.xnnpack.utils.configs import get_xnnpack_edge_compile_config
from executorch.exir import EdgeCompileConfig
from executorch.exir._serialize._flatbuffer import _flatc_decompile
from executorch.exir.backend.partitioner import Partitioner
from torch._export.pass_base import PassType
from torchao.quantization.pt2e.quantizer import Quantizer


class Export(BaseStages.Export):
    pass


class Quantize(BaseStages.Quantize):
    def __init__(
        self,
        quantizer: Optional[Quantizer] = None,
        quantization_config: Optional[QuantizationConfig] = None,
        calibrate: bool = True,
        calibration_samples: Optional[Sequence[Any]] = None,
        is_qat: Optional[bool] = False,
    ):
        super().__init__(
            quantizer=quantizer or XNNPACKQuantizer(),
            quantization_config=(
                quantization_config or get_symmetric_quantization_config(is_qat=is_qat)
            ),
            calibrate=calibrate,
            calibration_samples=calibration_samples,
            is_qat=is_qat,
        )


class RunPasses(BaseStages.RunPasses):
    def __init__(
        self,
        pass_list: Optional[List[Type[PassType]]] = None,
        pass_functions: Optional[List[Callable]] = None,
    ):
        super().__init__(
            pass_manager_cls=XNNPACKPassManager,
            pass_list=pass_list,
            pass_functions=pass_functions,
        )


class ToEdge(BaseStages.ToEdge):
    def __init__(self, edge_compile_config: Optional[EdgeCompileConfig] = None):
        super().__init__(edge_compile_config or get_xnnpack_edge_compile_config())


class ToEdgeTransformAndLower(BaseStages.ToEdgeTransformAndLower):
    def __init__(
        self,
        partitioners: Optional[List[Partitioner]] = None,
        edge_compile_config: Optional[EdgeCompileConfig] = None,
    ):
        super().__init__(
            default_partitioner_cls=XnnpackPartitioner,
            partitioners=partitioners,
            edge_compile_config=edge_compile_config
            or get_xnnpack_edge_compile_config(),
        )


class Partition(BaseStages.Partition):
    def __init__(self, partitioner: Optional[Partitioner] = None):
        super().__init__(
            partitioner=partitioner or XnnpackPartitioner(),
        )


class Serialize(BaseStages.Serialize):
    pass


class ToExecutorch(BaseStages.ToExecutorch):
    pass


class Tester(TesterBase):
    __test__ = False

    def __init__(
        self,
        module: torch.nn.Module,
        example_inputs: Tuple[torch.Tensor],
        dynamic_shapes: Optional[Tuple[Any]] = None,
        **kwargs,
    ):
        # Specialize for XNNPACK
        stage_classes = (
            executorch.backends.test.harness.Tester.default_stage_classes()
            | {
                StageType.EXPORT: Export,
                StageType.PARTITION: Partition,
                StageType.QUANTIZE: Quantize,
                StageType.RUN_PASSES: RunPasses,
                StageType.TO_EDGE: ToEdge,
                StageType.TO_EDGE_TRANSFORM_AND_LOWER: ToEdgeTransformAndLower,
                StageType.SERIALIZE: Serialize,
            }
        )

        super().__init__(
            module=module,
            stage_classes=stage_classes,
            example_inputs=example_inputs,
            dynamic_shapes=dynamic_shapes,
            **kwargs,
        )

    def xnnpack_node_types(self) -> Counter:
        """Node types in the serialized XNNPACK graph of the single delegate
        produced by to_edge_transform_and_lower."""
        graph_module = (
            self.get_artifact(StageType.TO_EDGE_TRANSFORM_AND_LOWER)
            .exported_program()
            .graph_module
        )
        (delegate,) = [
            node
            for node in graph_module.graph.nodes
            if node.target == torch.ops.higher_order.executorch_call_delegate
        ]
        payload = getattr(graph_module, delegate.args[0].target).processed_bytes
        header = XNNHeader.from_bytes(payload[: XNNHeader.EXPECTED_LENGTH])
        flatbuffer = payload[
            header.flatbuffer_offset : header.flatbuffer_offset + header.flatbuffer_size
        ]
        with tempfile.TemporaryDirectory() as d:
            schema_path = os.path.join(d, "schema.fbs")
            with open(schema_path, "wb") as f:
                f.write(_get_schema_bytes())
            bin_path = os.path.join(d, "graph.bin")
            with open(bin_path, "wb") as f:
                f.write(flatbuffer)
            _flatc_decompile(d, schema_path, bin_path)
            with open(os.path.join(d, "graph.json")) as f:
                graph = json.load(f)
        return Counter(node["xnode_union_type"] for node in graph["xnodes"])

    def check_xnnpack_node_count(self, expected: Dict[str, int]):
        node_type_names = {t.__name__ for t in get_args(XNodeUnion)}
        unknown = set(expected) - node_type_names
        assert not unknown, f"Unknown XNNPACK node types: {sorted(unknown)}"
        node_types = self.xnnpack_node_types()
        for node_type, count in expected.items():
            assert (
                node_types[node_type] == count
            ), f"Expected {count} {node_type} in the XNNPACK graph, got {node_types}"
        return self
