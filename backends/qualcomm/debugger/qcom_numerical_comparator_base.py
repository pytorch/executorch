# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import operator
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, final, Optional, Tuple

import executorch.exir as exir
import torch
from executorch.backends.qualcomm._passes.layout_transform import LayoutTransform
from executorch.backends.qualcomm.utils.constants import (
    QCOM_AXIS_ORDER,
    QCOM_QUANT_ATTRS,
    QCOM_SCALE,
    QCOM_ZERO_POINT,
)
from executorch.devtools.inspector.numerical_comparator import (
    IntermediateOutputMapping,
    NumericalComparatorBase,
)
from executorch.exir.sym_util import eval_shape

logger = logging.getLogger(__name__)


@dataclass
class NodeMetaInfo:
    node_name: str
    scale: Optional[float] = None
    zero_point: Optional[int] = None
    axis_order: Optional[Tuple[int, ...]] = None


class QcomNumericalComparatorBase(NumericalComparatorBase):
    """Base class for Qualcomm numerical comparators.

    This class locks down the `preprocessing` method to handle QNN-specific
    tensor transformations (dequantization, layout conversion) internally.
    Community users subclassing this base only need to implement `element_compare`.

    Attempting to override `preprocessing` in a subclass will raise TypeError
    at class definition time.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "preprocessing" in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} cannot override 'preprocessing'. "
                "Qualcomm handles preprocessing (dequantization, layout conversion) internally."
            )

    def __init__(self, edge_ep: exir.ExportedProgram) -> None:
        super().__init__()
        self.edge_ep = edge_ep

    @abstractmethod
    def metric_name(self) -> str:
        """
        A name for this metric evaluation.

        Returns:
            str: name of the metric evaluation.
        """
        ...

    @abstractmethod
    def is_valid_score(self, score: float) -> bool:
        """
        Determine whether a comparison score is within an acceptable range.

        Args:
            score: the comparison score to validate.

        Returns:
            bool: True if the score is acceptable, False otherwise.
        """
        ...

    @final
    def preprocessing(  # noqa: C901
        self, mapping: IntermediateOutputMapping
    ) -> IntermediateOutputMapping:

        def _preprocess_tensor(
            qnn_tensor: torch.Tensor, meta: NodeMetaInfo, golden_tensor: torch.Tensor
        ) -> torch.Tensor:
            if meta.scale is not None:
                # Dequantize
                qnn_tensor = (
                    qnn_tensor.to(torch.float32)
                    .sub(meta.zero_point)
                    .mul(meta.scale)
                    .contiguous()
                )
            if meta.axis_order:
                # QNN to Pytorch layout
                axis_order = LayoutTransform.get_axis_order(
                    eval_shape(qnn_tensor.shape), reverse=True
                )
                qnn_tensor = qnn_tensor.permute(axis_order)

            assert (
                golden_tensor.shape == qnn_tensor.shape
            ), f"{meta.node_name}'s golden and QNN tensor has different shape. Golden Tensor Shape: {golden_tensor.shape}. QNN Tensor Shape: {qnn_tensor.shape}."

            return qnn_tensor

        def _build_debug_handle_to_meta() -> (
            Dict[Tuple[int, ...], Dict[int, NodeMetaInfo]]
        ):
            debug_handle_to_meta: Dict[Tuple[int, ...], Dict[int, NodeMetaInfo]] = {}
            for node in self.edge_ep.graph_module.graph.nodes:
                if node.op != "call_function":
                    continue

                if (debug_handle := node.meta.get("debug_handle")) is None:
                    continue
                else:
                    debug_handle = (debug_handle,)

                quant_attrs = node.meta.get(QCOM_QUANT_ATTRS, {})
                node_meta_info = NodeMetaInfo(
                    node_name=node.name,
                    scale=quant_attrs.get(QCOM_SCALE, None),
                    zero_point=quant_attrs.get(QCOM_ZERO_POINT, None),
                    axis_order=node.meta.get(QCOM_AXIS_ORDER, None),
                )

                if any(user.target == operator.getitem for user in node.users):
                    # Assume if a node user is getitem, all users are getitem
                    assert all(
                        user.target == operator.getitem for user in node.users
                    ), "[QNN Delegate Debugger]: Expect all users to be getitem node"
                    continue

                # Multi-output op's getitem node shares the same debug handle.
                if node.target == operator.getitem:
                    output_idx = node.args[1]
                    debug_handle_to_meta.setdefault(debug_handle, {})[
                        output_idx
                    ] = node_meta_info
                else:
                    assert (
                        debug_handle not in debug_handle_to_meta
                    ), f"[QNN Delegate Debugger]: Duplicate handle_id {debug_handle} found when visiting {node.name}."
                    debug_handle_to_meta[debug_handle] = {0: node_meta_info}

            return debug_handle_to_meta

        debug_handle_to_meta = _build_debug_handle_to_meta()
        processed_mapping: IntermediateOutputMapping = {}
        for (golden_handle, golden_output), (qnn_handle, qnn_output) in mapping.items():
            assert (
                golden_handle == qnn_handle
            ), f"Expecting the handle to match, aot handle: {golden_handle}, qnn_handle: {qnn_handle}."
            if node_meta_dict := debug_handle_to_meta.get(qnn_handle, None):
                if isinstance(qnn_output, tuple):
                    assert len(qnn_output) <= len(
                        node_meta_dict
                    ), f"node_meta has {len(node_meta_dict)} entries but qnn_output has {len(qnn_output)} elements."
                    if len(node_meta_dict) != len(qnn_output):
                        logging.warning(
                            f"Number of QNN output {len(qnn_output)} mismatched with number of output for edge module {len(node_meta_dict)}. This is possibly due to multi-outputs and QNN does not use all outputs. Please verify the following meta from edge module and ensure this is desired: {node_meta_dict}."
                        )

                    processed = []
                    for idx, q_tensor in enumerate(qnn_output):
                        processed.append(
                            _preprocess_tensor(
                                qnn_tensor=q_tensor,
                                meta=node_meta_dict[idx],
                                golden_tensor=golden_output[idx],
                            )
                        )

                    qnn_output = tuple(processed)
                else:
                    assert (
                        len(node_meta_dict) == 1 and 0 in node_meta_dict
                    ), f"Single output expected node_meta_dict with key 0, got keys {list(node_meta_dict.keys())}"
                    qnn_output = _preprocess_tensor(
                        qnn_tensor=qnn_output,
                        meta=node_meta_dict[0],
                        golden_tensor=golden_output,
                    )

            processed_mapping[(golden_handle, golden_output)] = (qnn_handle, qnn_output)
        return processed_mapping
