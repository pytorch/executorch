# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
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

from nncf.quantization.algorithms.weight_compression.weight_lowering import (  # type: ignore[import-untyped]
    do_integer_quantization,
)
from nncf.tensor.tensor import Tensor  # type: ignore[import-untyped]
from nncf.torch.graph.transformations.commands import (  # type: ignore[import-untyped]
    PTTargetPoint,
    TargetType,
)
from nncf.torch.quantization.layers import (  # type: ignore[import-untyped]
    INT4AsymmetricWeightsDecompressor,
    INT4SymmetricWeightsDecompressor,
    INT8AsymmetricWeightsDecompressor,
    INT8SymmetricWeightsDecompressor,
)
from torchao.quantization.observer import AffineQuantizedMinMaxObserver
from torchao.quantization.pt2e import (
    get_block_size,
    MappingType,
    PerAxis,
    PerChannelMinMaxObserver,
    PerGroup,
)
from torchao.quantization.quant_primitives import _get_reduction_params


class PTPerBlockParamObserver(AffineQuantizedMinMaxObserver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        qmode = (
            CompressWeightsMode.INT4_ASYM
            if self.mapping_type == MappingType.ASYMMETRIC
            else CompressWeightsMode.INT4_SYM
        )
        assert isinstance(
            self.granularity, PerGroup
        ), "Only PerGroup granularity is supported"
        self.wc_config = WeightCompressionConfig(
            mode=qmode, group_size=self.granularity.group_size
        )

    def calculate_qparams(self, weight):
        assert hasattr(self, "min_val") and hasattr(
            self, "max_val"
        ), "Expecting the observer has min_val and max_val, please run the observer before calling calculate_qparams"
        _, reduction_dims = _get_reduction_params(self.block_size, weight.size())
        assert len(reduction_dims) == 1, "Only 1-D group size is supported"
        reduction_dims = reduction_dims[0] - 1
        q_weight, scale, zp = do_integer_quantization(
            Tensor(weight), self.wc_config, reduction_axes=reduction_dims
        )
        zp = zp.data if zp is not None else None
        return q_weight.data, scale.data, zp

    def convert(self, model: torch.fx.GraphModule, observer_node: torch.fx.Node):
        print("calling convert")
        assert (
            self.original_dtype is not None
        ), "Expecting original_dtype to be populated"
        weight_node = observer_node.args[0]
        original_weight = get_tensor_constant_from_node(weight_node, model)
        q_weight, scale, zero_point = self.calculate_qparams(original_weight)

        with model.graph.inserting_before(observer_node):
            if zero_point is not None:
                decompressor = INT4AsymmetricWeightsDecompressor(
                    scale,
                    zero_point,
                    q_weight.shape,
                    original_weight.shape,
                    original_weight.dtype,
                )
            else:
                decompressor = INT4SymmetricWeightsDecompressor(
                    scale, q_weight.shape, original_weight.shape, original_weight.dtype
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


class NNCFInt8observer(PerChannelMinMaxObserver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        qmode = (
            CompressWeightsMode.INT8_SYM
            if self.qscheme == torch.per_channel_symmetric
            else CompressWeightsMode.INT8_ASYM
        )
        self.wc_config = WeightCompressionConfig(mode=qmode)

    def calculate_qparams(self, weight):
        assert hasattr(self, "min_val") and hasattr(
            self, "max_val"
        ), "Expecting the observer has min_val and max_val, please run the observer before calling calculate_qparams"
        self.granularity = PerAxis(axis=self.ch_axis)
        self.block_size = get_block_size(weight.shape, self.granularity)
        _, reduction_dims = _get_reduction_params(self.block_size, weight.size())
        q_weight, scale, zp = do_integer_quantization(
            Tensor(weight), self.wc_config, reduction_axes=reduction_dims
        )
        zp = zp.data if zp is not None else None
        return q_weight.data, scale.data, zp

    def convert(self, model: torch.fx.GraphModule, observer_node: torch.fx.Node):
        print("calling convert")
        weight_node = observer_node.args[0]
        original_weight = get_tensor_constant_from_node(weight_node, model)
        q_weight, scale, zero_point = self.calculate_qparams(original_weight)

        with model.graph.inserting_before(observer_node):
            if zero_point is not None:
                decompressor = INT8AsymmetricWeightsDecompressor(
                    scale, zero_point, original_weight.dtype
                )
            else:
                decompressor = INT8SymmetricWeightsDecompressor(
                    scale, original_weight.dtype
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
