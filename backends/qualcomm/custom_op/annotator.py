# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Union

import torch
from executorch.backends.qualcomm.quantizer.rules import _is_float_tensor
from torchao.quantization.pt2e.quantizer import (
    QuantizationAnnotation,
    QuantizationSpec,
    SharedQuantizationSpec,
)
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY

logger = logging.getLogger(__name__)


@dataclass
class IOQuantConfig:
    """
    Quantization config for custom op inputs and outputs.

    Attributes:
        input_quant_specs: Maps input index to its QuantizationSpec.
            Only indices present in the dict are annotated. If None, no inputs
            are annotated.
        output_quant_specs: Maps output index to its QuantizationSpec.
            For single-output ops annotation is done on the op node. For multi-output ops,
            each index corresponds to a downstream getitem user. If None, no
            outputs are annotated.
    """

    input_quant_specs: Optional[
        Dict[int, Union[QuantizationSpec, SharedQuantizationSpec]]
    ] = None
    output_quant_specs: Optional[
        Dict[int, Union[QuantizationSpec, SharedQuantizationSpec]]
    ] = None


class CustomOpsQuantAnnotator:
    """
    Holds op IOQuantConfigs and builds a single annotation function
    compatible with make_quantizer(custom_annotations=...).
    """

    def __init__(self):
        self._registry: Dict = {}  # {op_target: IOQuantConfig}

    def register_annotation(
        self,
        op_target,
        io_quant_config: IOQuantConfig,
    ) -> "CustomOpsQuantAnnotator":
        """
        Register quantization config for custom op.

        Args:
            op_target: The torch op target (e.g. torch.ops.my_ops.custom_op.default).
            io_quant_config: IOQuantConfig specifying how to quantize inputs and outputs.

        Returns self for method chaining.
        """
        self._registry[op_target] = io_quant_config
        return self

    def build_annotation_fn(self) -> Callable[[torch.fx.GraphModule], None]:
        """
        Build and return an annotation function for all registered ops.

        The returned function has signature (gm: GraphModule) -> None and
        can be passed directly to make_quantizer(custom_annotations=(fn,)).
        """
        registry = dict(self._registry)

        def annotate_custom_ops(gm: torch.fx.GraphModule) -> None:
            for node in gm.graph.nodes:
                if node.target not in registry:
                    continue

                cfg = registry[node.target]
                input_qspec_map = {}
                if cfg.input_quant_specs is not None:
                    for arg_idx, spec in cfg.input_quant_specs.items():
                        if arg_idx >= len(node.args):
                            raise ValueError(
                                f"IOQuantConfig error for '{node.name}' ({node.target}): "
                                f"input_quant_specs index {arg_idx} is out of range "
                                f"(op has {len(node.args)} args)"
                            )
                        if not _is_float_tensor(node.args[arg_idx]):
                            logger.debug(
                                f"Skipping quantization of input {arg_idx} for "
                                f"'{node.name}' ({node.target}): expected a float tensor."
                            )
                            continue
                        logger.debug(
                            f"Annotating input {arg_idx} of '{node.name}' ({node.target}) "
                            f"with {spec}"
                        )
                        input_qspec_map[node.args[arg_idx]] = spec

                if not cfg.output_quant_specs or len(cfg.output_quant_specs) <= 1:
                    # Single output — annotate on the op node
                    output_spec = (
                        cfg.output_quant_specs.get(0)
                        if cfg.output_quant_specs
                        else None
                    )
                    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                        input_qspec_map=input_qspec_map,
                        output_qspec=output_spec,
                        _annotated=True,
                    )
                else:
                    # Tuple output — push quantization down to getitem users
                    node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                        input_qspec_map=input_qspec_map,
                        output_qspec=None,
                        _annotated=True,
                    )
                    for user in node.users:
                        output_idx = user.args[1]
                        spec = cfg.output_quant_specs.get(output_idx)

                        if spec is not None:
                            user.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                                output_qspec=spec,
                                _annotated=True,
                            )

        return annotate_custom_ops
