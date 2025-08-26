# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file found in the
# LICENSE file in the root directory of this source tree.

# mypy: disable-error-code=import-not-found

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import nncf.torch.graph.operator_metatypes as om  # type: ignore[import-untyped]

import torch
from nncf.experimental.torch.fx.nncf_graph_builder import (  # type: ignore[import-untyped]
    GraphConverter,
)

from nncf.experimental.torch.fx.node_utils import (  # type: ignore[import-untyped]
    get_tensor_constant_from_node,
)
from nncf.experimental.torch.fx.transformations import (  # type: ignore[import-untyped]
    constant_update_fn,
    module_insertion_transformation_builder,
)
from nncf.parameters import CompressWeightsMode  # type: ignore[import-untyped]
from nncf.quantization.algorithms.weight_compression.config import (  # type: ignore[import-untyped]
    WeightCompressionConfig,
)
from nncf.quantization.algorithms.weight_compression.torch_fx_backend import (  # type: ignore[import-untyped]
    FXWeightCompressionAlgoBackend,
)
from nncf.quantization.algorithms.weight_compression.weight_lowering import (  # type: ignore[import-untyped]
    do_integer_quantization,
)
from nncf.tensor.tensor import Tensor  # type: ignore[import-untyped]
from nncf.torch.graph.transformations.commands import (  # type: ignore[import-untyped]
    PTTargetPoint,
    TargetType,
)
from nncf.torch.quantization.layers import (  # type: ignore[import-untyped]
    BaseWeightsDecompressor,
    INT4AsymmetricWeightsDecompressor,
    INT4SymmetricWeightsDecompressor,
    INT8AsymmetricWeightsDecompressor,
    INT8SymmetricWeightsDecompressor,
)
from torchao.quantization.pt2e import MappingType, ObserverBase
from nncf.torch.model_graph_manager import get_weight_compression_reduction_axes

class WeightObserverBase(ObserverBase, ABC):
    """
    Base implementation of an NNCF observer that defines the rules for compressing layer weights into the OpenVINO representation.
    """

    def calculate_qparams(  # type: ignore[override]
        self,
        weight: torch.Tensor,
        observer_node: torch.fx.Node,
        model: torch.fx.GraphModule,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Calculate quantization parameters such as scale, quantized weight and zero point.

        :param weight: FP weight to be used for calculating qparams.
        :return: quantization params quantized weight, scale and zero point
        """
        ndims = len(weight.size())
        node_with_weight, weight_port_id = (
            WeightObserverBase.get_node_with_weight_and_port_ids(observer_node, model)
        )
        _, node_metatype = GraphConverter.get_node_type_and_metatype(
            node_with_weight, model
        )
        # Special case where embedding metatype has to be mapped to AtenEmbedding metatype
        node_metatype = (
            om.PTAtenEmbeddingMetatype
            if node_metatype == om.PTEmbeddingMetatype
            else node_metatype
        )
        reduction_dims = get_weight_compression_reduction_axes(
            node_metatype, weight_port_id, ndims
        )
        reduction_dims = tuple(reduction_dims)

        q_weight, scale, zp = do_integer_quantization(
            Tensor(weight), self.wc_config, reduction_axes=reduction_dims
        )
        zp = zp.data if zp is not None else None
        return q_weight.data, scale.data, zp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def get_node_with_weight_and_port_ids(
        observer_node: torch.fx.Node, model: torch.fx.GraphModule
    ) -> Tuple[torch.fx.Node, int]:
        """
        Returns the node which contains the weight and the weight port id.

        :param observer_node: Observer node for the weight.
        :param graph: The model.
        :return: Node which contains the weight (for eg. Linear node) and the port ID for the weight.
        """
        for node in model.graph.nodes:
            if observer_node in node.all_input_nodes:
                return node, node.all_input_nodes.index(observer_node)
        msg = f"Observer node {observer_node.name} has no consumer node"
        raise RuntimeError(msg)

    def convert(
        self, model: torch.fx.GraphModule, observer_node: torch.fx.Node
    ) -> None:
        """
        Converts the weight observer node into a decompression subgraph after calibration.
        This method is responsible for transforming the model after the quantization preparation
        and calibration phases. It replaces the observer node with the quantized weight and a decompression
        module.

        :param model: A `torch.fx.GraphModule` representing the statically traced model
                    with observer nodes attached and calibrated.
        :param observer_node: The `torch.fx.Node` corresponding to the observer module for
                            the weight that is being transformed into a compressed representation.
        """
        weight_node = observer_node.args[0]
        original_weight = get_tensor_constant_from_node(weight_node, model)
        q_weight, scale, zero_point = self.calculate_qparams(
            original_weight, observer_node, model
        )

        decompressor = self._create_decompressor(
            scale, zero_point, q_weight, original_weight
        )
        packed_q_weight = decompressor.pack_weight(q_weight)

        constant_update_fn(model, observer_node, packed_q_weight, input_port_id=0)

        compressed_weight_name = observer_node.all_input_nodes[0].name
        decompressor_suffix = "_".join(
            compressed_weight_name.replace(".", "_").split("_")[:-2]
        )
        decompressor_name = f"{decompressor.quantization_mode}_weights_decompressor_{decompressor_suffix}"

        module_insertion_transformation_builder(
            decompressor,
            [
                PTTargetPoint(
                    TargetType.OPERATOR_POST_HOOK,
                    target_node_name=compressed_weight_name,
                )
            ],
            decompressor_name,
        )(model)

        decomp_node = observer_node.args[0]
        observer_node.replace_all_uses_with(decomp_node)  # type: ignore[arg-type]
        model.graph.erase_node(observer_node)

    @abstractmethod
    def _create_decompressor(
        self,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        q_weight: torch.Tensor,
        original_weight: torch.Tensor,
    ) -> BaseWeightsDecompressor:
        """
        Used to return the respective NNCF decompressor for different types of quantization.

        :param scale: Calculated scale quantization parameter.
        :param zero_point: Calculated zero_point quantization parameter.
        :param q_weight: Calculated quantized weight.
        :param original_weight: FP weight.
        :return: NNCF observer according to the qmode which creates the decompression subgraph supported by OpenVINO.
        """
        pass

    @abstractmethod
    def get_wc_config(self) -> WeightCompressionConfig:
        """
        Used to return the respective NNCF Weight Compression Config.

        :return: Weight compression config with the compression information such as qmode, group_size etc.
        """
        pass


class INT4WeightObserver(WeightObserverBase):
    """
    This class defines the behavior for INT4 Weight Compression which has per-group granularity.
    """

    def __init__(
        self,
        group_size: int,
        mapping_type: MappingType,
        target_dtype: torch.dtype,
        *args,
        **kwargs,
    ) -> None:
        """
        :param group_size: Group size for group wise quantization. group_size=-1 means it is per-channel quantization.
        :param mapping_type: MappingType.SYMMETRIC and MappingType.ASYMMETRIC are supported types for this argument for symmetric or asymmetric quantization.
        :param target_dtype: target dtype for quantization such as int8, uint8, etc.
        """
        super().__init__(dtype=target_dtype, is_dynamic=False)
        self.wc_config = None
        self.mapping_type = mapping_type

        qmode = (
            CompressWeightsMode.INT4_ASYM
            if self.mapping_type == MappingType.ASYMMETRIC
            else CompressWeightsMode.INT4_SYM
        )
        self.wc_config = WeightCompressionConfig(mode=qmode, group_size=group_size)

    def _create_decompressor(
        self,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        q_weight: torch.Tensor,
        original_weight: torch.Tensor,
    ) -> BaseWeightsDecompressor:
        if zero_point is not None:
            return INT4AsymmetricWeightsDecompressor(
                scale,
                zero_point,
                q_weight.shape,
                original_weight.shape,
                original_weight.dtype,
            )
        else:
            return INT4SymmetricWeightsDecompressor(
                scale, q_weight.shape, original_weight.shape, original_weight.dtype
            )

    def get_wc_config(self):
        return self.wc_config


class INT8WeightObserver(WeightObserverBase):
    """
    This class defines the behavior for Int8 WC which has per channel granularity.
    """

    def __init__(
        self,
        qscheme: torch.qscheme,
        dtype: torch.dtype,
        ch_axis: int = 0,
        *args,
        **kwargs,
    ) -> None:
        """
        :param qscheme: Quantization scheme which is per-channel for Int8 WC.
        :param dtype: dtype for quantization such as int8, uint8, etc..
        :param ch_axis: Channel axis.
        """
        super().__init__(dtype=dtype, is_dynamic=False)
        self.wc_config = None
        self.qscheme = qscheme

        qmode = (
            CompressWeightsMode.INT8_SYM
            if self.qscheme == torch.per_channel_symmetric
            else CompressWeightsMode.INT8_ASYM
        )
        self.wc_config = WeightCompressionConfig(mode=qmode)

    def _create_decompressor(
        self,
        scale: torch.Tensor,
        zero_point: Optional[torch.Tensor],
        q_weight: torch.Tensor,
        original_weight: torch.Tensor,
    ) -> BaseWeightsDecompressor:
        if zero_point is not None:
            return INT8AsymmetricWeightsDecompressor(
                scale, zero_point, original_weight.dtype
            )
        else:
            return INT8SymmetricWeightsDecompressor(scale, original_weight.dtype)

    def get_wc_config(self):
        return self.wc_config