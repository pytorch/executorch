# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import numbers
import operator
from functools import partial
from typing import Dict, List, Optional, Sequence, Tuple

import executorch.backends.qualcomm.builders.qnn_constants as QnnConstants
import torch

from executorch.backends.qualcomm.quantizer.observers.concat_observer import (
    ConcatObserver,
)
from executorch.backends.qualcomm.quantizer.qconfig import (
    get_16a16w_qnn_ptq_config,
    get_16a4w_qnn_qat_config,
    get_8a8w_qnn_qat_config,
    QuantizationConfig,
)
from executorch.backends.qualcomm.quantizer.rules import (
    _is_annotated,
    _is_float_tensor,
    _mark_nodes_as_annotated,
    annotate_binary,
    annotate_conv,
    annotate_in_out_obs_sharing_op,
    annotate_single_in,
    annotate_single_in_share_out,
    GeneralOpDef,
    OpQuantRule,
    Q_ANNOTATION_KEY,
    validate_against_backend_constraints,
)
from executorch.backends.qualcomm.quantizer.validators import (
    ConstraintCache,
    NormalizedConstraints,
)
from executorch.backends.qualcomm.serialization.qc_schema import HtpArch, SocInfo
from executorch.backends.qualcomm.utils.constants import QCOM_BLOCK_SIZE
from torch._ops import OpOverload
from torch.fx import Node
from torchao.quantization.pt2e import FixedQParamsFakeQuantize, FixedQParamsObserver
from torchao.quantization.pt2e.quantizer import (
    annotate_input_qspec_map,
    annotate_output_qspec,
    DerivedQuantizationSpec,
    QuantizationAnnotation,
    QuantizationSpec,
    SharedQuantizationSpec,
)

_RULES: Dict[OpOverload, OpQuantRule] = {}
_CONSTRAINT_CACHE: ConstraintCache = ConstraintCache()


def get_rules() -> Dict[str, List[OpQuantRule]]:
    """
    Entry point for registry_loader.load_backend_rules("htp").
    """
    return _RULES


def get_constraint_cache() -> ConstraintCache:
    return _CONSTRAINT_CACHE


def register_annotator(aten_ops: List[OpOverload], qnn_op: Optional[str]):
    def _wrap(op_def: GeneralOpDef):
        for aten_op in aten_ops:
            annotate_fn = op_def.annotate
            validate_fn = op_def.validate
            rule = OpQuantRule(
                aten_op=aten_op,
                qnn_op=qnn_op,
                annotate_fn=annotate_fn,
                validate_fn=validate_fn,
            )
            _RULES[rule.aten_op] = rule
        return op_def

    return _wrap


def validate_lpbq_support(soc_info: SocInfo) -> bool:
    valid = soc_info.htp_info.htp_arch >= HtpArch.V69
    return valid


def validate_16a16w_support(soc_info: SocInfo) -> bool:
    valid = soc_info.htp_info.htp_arch >= HtpArch.V73
    return valid


@register_annotator([torch.ops.aten.abs.default], QnnConstants.OpElementWiseAbs.op_name)
class Abs(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.add, torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor],
    QnnConstants.OpElementWiseAdd.op_name,
)
class Add(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.amax.default], QnnConstants.OpReduceMax.op_name)
class Amax(GeneralOpDef):
    pass


@register_annotator([torch.ops.aten.amin.default], QnnConstants.OpReduceMin.op_name)
class Amin(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.__and__.Tensor, torch.ops.aten.logical_and.default],
    QnnConstants.OpElementWiseAnd.op_name,
)
class And(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.argmax.default], QnnConstants.OpArgmax.op_name)
class Argmax(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_single_in(node, quantization_config)


@register_annotator([torch.ops.aten.argmin.default], QnnConstants.OpArgmin.op_name)
class Argmin(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_single_in(node, quantization_config)


@register_annotator(
    [torch.ops.aten.asin.default], QnnConstants.OpElementWiseAsin.op_name
)
class Asin(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.atan.default], QnnConstants.OpElementWiseAtan.op_name
)
class Atan(GeneralOpDef):
    pass


@register_annotator(
    [
        torch.ops.aten.adaptive_avg_pool1d.default,
        torch.ops.aten.adaptive_avg_pool2d.default,
        torch.ops.aten.avg_pool1d.default,
        torch.ops.aten.avg_pool2d.default,
    ],
    QnnConstants.OpPoolAvg2d.op_name,
)
class AvgPool2d(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.avg_pool3d.default, torch.ops.aten.adaptive_avg_pool3d.default],
    QnnConstants.OpPoolAvg3d.op_name,
)
class AvgPool3d(GeneralOpDef):
    pass


# TODO: Batch_norm op cannot directly map to QNN OpBatchnorm due to the number of input doesn't match.
@register_annotator(
    [torch.ops.aten.batch_norm.default, torch.ops.aten.instance_norm.default],
    qnn_op=None,
)
class BatchNorm(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        act, weight, bias = node.args[0:3]
        if _is_annotated([node]):
            return

        annotated_args = [act]
        annotate_input_qspec_map(
            node,
            act,
            quantization_config.input_activation,
        )
        # QNN requires uint8 instead of int8 in 'weight' config
        if weight is not None:
            annotate_input_qspec_map(
                node,
                weight,
                quantization_config.input_activation,
            )
            annotated_args.append(weight)

        if bias is not None:
            annotate_input_qspec_map(
                node,
                bias,
                quantization_config.bias,
            )
            annotated_args.append(bias)

        annotate_output_qspec(node, quantization_config.output_activation)
        _mark_nodes_as_annotated([node, *annotated_args])


@register_annotator([torch.ops.aten.to.dtype], QnnConstants.OpCast.op_name)
class Cast(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.cat.default, torch.ops.aten.concat.default],
    QnnConstants.OpConcat.op_name,
)
class Cat(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        if _is_annotated([node]) or not _is_float_tensor(node):
            return

        input_qspec_map, input_nodes = {}, node.args[0]
        for input in input_nodes:
            input_qspec = input.meta.get(Q_ANNOTATION_KEY, None)
            qspec = getattr(input_qspec, "output_qspec", None)
            # keep shared qspec here for propagation the data range
            # without introducing extra requantizations
            if isinstance(qspec, SharedQuantizationSpec):
                input_qspec_map[input] = SharedQuantizationSpec(input)
            else:
                input_qspec_map[input] = quantization_config.input_activation

        output_qspec = QuantizationSpec(
            dtype=quantization_config.output_activation.dtype,
            qscheme=quantization_config.output_activation.qscheme,
            quant_max=quantization_config.output_activation.quant_max,
            quant_min=quantization_config.output_activation.quant_min,
            observer_or_fake_quant_ctr=ConcatObserver.with_args(
                # we need to know the concat node in order to hack all the input observers' data range
                # since deep copy of fake tensor (node.meta["val"]) is inhibited
                # we could only ship grap & node name and perform postprocess inside observer currently
                **{
                    "node_name": node.name,
                    "graph": node.graph,
                }
            ),
        )
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_qspec,
            _annotated=True,
        )


@register_annotator([torch.ops.aten.cdist.default], qnn_op=None)
class Cdist(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)


@register_annotator(
    [torch.ops.aten.ceil.default], QnnConstants.OpElementWiseCeil.op_name
)
class Ceil(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.channel_shuffle.default], QnnConstants.OpChannelShuffle.op_name
)
class ChannelShuffle(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_in_out_obs_sharing_op(node, quantization_config)
        if not _is_annotated([node]):
            annotate_single_in_share_out(node, quantization_config)


@register_annotator(
    [
        torch.ops.aten.split_with_sizes.default,
        torch.ops.aten.split.Tensor,
        torch.ops.aten.chunk.default,
    ],
    QnnConstants.OpSplit.op_name,
)
class Chunk(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        if _is_annotated([node]):
            return

        input_qspec_map = {}
        input_act = node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = quantization_config.input_activation
        share_qparams_with_input_node_qspec = SharedQuantizationSpec((input_act, node))

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            _annotated=True,
        )

        for user in node.users:
            user.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                output_qspec=share_qparams_with_input_node_qspec,
                _annotated=True,
            )


@register_annotator(
    [torch.ops.aten.col2im.default, torch.ops.aten.im2col.default], qnn_op=None
)
class ColIm(GeneralOpDef):
    pass


@register_annotator(
    [
        torch.ops.aten.arange.default,
        torch.ops.aten.arange.start,
        torch.ops.aten.arange.start_step,
        torch.ops.aten.scalar_tensor.default,
        torch.ops.aten.full_like.default,
        torch.ops.aten.full.default,
        torch.ops.aten.zeros.default,
        torch.ops.aten.zeros_like.default,
        torch.ops.aten.ones.default,
        torch.ops.aten.ones_like.default,
    ],
    qnn_op=None,
)
class ConstantOp(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        if _is_annotated([node]) or not _is_float_tensor(node):
            return

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map={},
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )


@register_annotator(
    [
        torch.ops.aten.conv1d.default,
        torch.ops.aten.conv2d.default,
        torch.ops.aten.conv2d.padding,
        torch.ops.aten.convolution.default,
    ],
    QnnConstants.OpConv2d.op_name,
)
class Conv2d(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_conv(node, quantization_config)

    @staticmethod
    def validate(
        node: Node, constraints_list: List[NormalizedConstraints], soc_info: SocInfo
    ) -> bool:
        valid = True
        if not _is_annotated([node]):
            return valid

        weight_node = node.args[1]
        weight_qspec = node.meta[Q_ANNOTATION_KEY].input_qspec_map.get(
            weight_node, None
        )
        if (
            weight_qspec
            and weight_qspec.observer_or_fake_quant_ctr.p.keywords.get(
                QCOM_BLOCK_SIZE, None
            )
            is not None
        ):
            valid &= validate_lpbq_support(soc_info)
            if not valid:
                logging.warning(
                    f"LPBQ (16a4w block-wise quantization) requires V69 or newer for {node.name}"
                )

        act_node = node.args[0]
        act_qspec = node.meta[Q_ANNOTATION_KEY].input_qspec_map.get(act_node, None)
        if (
            act_qspec
            and act_qspec.dtype == torch.int32
            and weight_qspec
            and weight_qspec.dtype == torch.int32
        ):
            valid &= validate_16a16w_support(soc_info)
            if not valid:
                logging.warning(
                    f"16-bit activations + 16-bit weights requires V73 or newer for {node.name}"
                )

        valid &= validate_against_backend_constraints(node, constraints_list)
        return valid


@register_annotator(
    [
        torch.ops.aten.conv3d.default,
    ],
    QnnConstants.OpConv3d.op_name,
)
class Conv3d(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_conv(node, quantization_config)

    @staticmethod
    def validate(
        node: Node, constraints_list: List[NormalizedConstraints], soc_info: SocInfo
    ) -> bool:
        valid = True
        if not _is_annotated([node]):
            return valid

        weight_node = node.args[1]
        weight_qspec = node.meta[Q_ANNOTATION_KEY].input_qspec_map.get(
            weight_node, None
        )
        if (
            weight_qspec
            and weight_qspec.observer_or_fake_quant_ctr.p.keywords.get(
                QCOM_BLOCK_SIZE, None
            )
            is not None
        ):
            valid &= validate_lpbq_support(soc_info)
            if not valid:
                logging.warning(
                    f"LPBQ (16a4w block-wise quantization) requires V69 or newer for {node.name}"
                )
        valid &= validate_against_backend_constraints(node, constraints_list)
        return valid


@register_annotator([torch.ops.aten.cos.default], QnnConstants.OpElementWiseCos.op_name)
class Cos(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.cumsum.default], QnnConstants.OpCumulativeSum.op_name
)
class CumSum(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.div, torch.ops.aten.div.Tensor, torch.ops.aten.divide.Tensor],
    QnnConstants.OpElementWiseDivide.op_name,
)
class Div(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        def _derived_inp1_const_div_quant_spec(
            node: torch.fx.Node, output_qspec: QuantizationSpec
        ) -> DerivedQuantizationSpec:
            def _derive_div_qparams_fn(
                obs_or_fqs: List,
                const_val: float,
            ) -> Tuple[torch.Tensor, torch.Tensor]:
                inp_0_obs_or_fq = obs_or_fqs[0]
                inp_0_scale, inp_0_zp = inp_0_obs_or_fq.calculate_qparams()
                derived_scale = inp_0_scale / const_val
                return (derived_scale, inp_0_zp)

            inp_0 = node.args[0]
            const_inp_1 = node.args[1]
            _derive_div_qparams_with_const_fn = partial(
                _derive_div_qparams_fn, const_val=const_inp_1
            )

            q_min = (
                torch.iinfo(output_qspec.dtype).min
                if output_qspec.quant_min is None
                else output_qspec.quant_min
            )
            q_max = (
                torch.iinfo(output_qspec.dtype).max
                if output_qspec.quant_max is None
                else output_qspec.quant_max
            )
            return DerivedQuantizationSpec(
                derived_from=[(inp_0, node)],
                derive_qparams_fn=_derive_div_qparams_with_const_fn,
                dtype=output_qspec.dtype,
                quant_min=q_min,
                quant_max=q_max,
                ch_axis=0,
                qscheme=output_qspec.qscheme,
            )

        if [a for a in node.args if isinstance(a, Node)]:
            annotate_binary(node, quantization_config)
        # special constant divisor case
        elif isinstance(node.args[0], Node) and isinstance(
            node.args[1], numbers.Number
        ):
            if _is_annotated([node]):
                return

            input_act_qspec = quantization_config.input_activation
            output_act_qspec = _derived_inp1_const_div_quant_spec(
                node, quantization_config.output_activation
            )
            input_qspec_map = {}
            input_act0 = node.args[0]
            if _is_float_tensor(input_act0):
                input_qspec_map[input_act0] = input_act_qspec

            node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=output_act_qspec,
                _annotated=True,
            )
        else:
            raise NotImplementedError(f"No quant annotation is implemented for {node}.")


@register_annotator([torch.ops.aten.elu.default], QnnConstants.OpElu.op_name)
class Elu(GeneralOpDef):
    pass


# TODO: Embedding op cannot directly map to OpGather because the index input in torch is not a tensor.
@register_annotator(
    [
        torch.ops.aten.gather.default,
        torch.ops.aten.index.Tensor,
        torch.ops.aten.index_select.default,
    ],
    qnn_op=None,
)
class Gather(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        # args[2] = indices, which should be int
        annotate_in_out_obs_sharing_op(node, quantization_config)
        if not _is_annotated([node]):
            annotate_single_in_share_out(node, quantization_config)


@register_annotator(
    [
        torch.ops.aten.embedding.default,
    ],
    qnn_op=None,
)
class Embedding(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        weight = node.args[0]

        # Only quantize if input is a float tensor
        if _is_annotated([node]) or not _is_float_tensor(weight):
            return

        is_pcq_embedding = quantization_config.per_channel_embedding
        input_qspec_map = {}
        input_qspec_map[weight] = (
            quantization_config.weight
            if is_pcq_embedding
            else quantization_config.input_activation
        )
        output_qspec = (
            quantization_config.input_activation
            if is_pcq_embedding
            else SharedQuantizationSpec((weight, node))
        )
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_qspec,
            _annotated=True,
        )


@register_annotator([torch.ops.aten.eq.Tensor], QnnConstants.OpElementWiseEqual.op_name)
class Equal(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.exp.default], QnnConstants.OpElementWiseExp.op_name)
class Exp(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.floor.default], QnnConstants.OpElementWiseFloor.op_name
)
class Floor(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.floor_divide.default], QnnConstants.OpElementWiseFloorDiv.op_name
)
class FloorDivide(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.gelu.default], QnnConstants.OpGelu.op_name)
class Gelu(GeneralOpDef):
    pass


@register_annotator([operator.getitem], qnn_op=None)
class GetItem(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        if _is_annotated([node]) or not _is_float_tensor(node):
            return

        out_act_quantization_spec = quantization_config.output_activation
        # QNN constraint, topk output_0 requires having the same quant config as input
        if node.args[0].target == torch.ops.aten.topk.default:
            out_act_quantization_spec = SharedQuantizationSpec(node.args[0])
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            output_qspec=out_act_quantization_spec,
            _annotated=True,
        )


@register_annotator(
    [torch.ops.aten.gt.Tensor], QnnConstants.OpElementWiseGreater.op_name
)
class Greater(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)


@register_annotator(
    [torch.ops.aten.ge.Tensor], QnnConstants.OpElementWiseGreaterEqual.op_name
)
class GreaterEqual(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)


@register_annotator(
    [torch.ops.aten.grid_sampler.default], QnnConstants.OpGridSample.op_name
)
class GridSample(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)


@register_annotator(
    [torch.ops.aten.group_norm.default], QnnConstants.OpGroupNorm.op_name
)
class GroupNorm(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        act_node = node.args[0]
        weight_node = node.args[2]
        bias_node = None
        if len(node.args) > 2:
            bias_node = node.args[3]

        if _is_annotated([node]):
            return

        annotate_input_qspec_map(
            node,
            act_node,
            quantization_config.input_activation,
        )
        annotate_input_qspec_map(
            node,
            weight_node,
            quantization_config.weight,
        )
        nodes_to_mark_annotated = [node, weight_node]
        if bias_node:
            annotate_input_qspec_map(
                node,
                bias_node,
                quantization_config.bias,
            )
            nodes_to_mark_annotated.append(bias_node)
        annotate_output_qspec(node, quantization_config.output_activation)
        _mark_nodes_as_annotated(nodes_to_mark_annotated)


@register_annotator(
    [torch.ops.aten.hardsigmoid.default, torch.ops.aten.hardsigmoid_.default],
    QnnConstants.OpElementWiseNeuron.op_name,
)
class HardSigmoid(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        if _is_annotated([node]):
            return

        input_qspec_map = {}
        input_act = node.args[0]
        input_qspec_map[input_act] = quantization_config.input_activation

        assert isinstance(input_act, Node)
        out_qconf = quantization_config.output_activation

        q_max = (
            torch.iinfo(out_qconf.dtype).max
            if out_qconf.quant_max is None
            else out_qconf.quant_max
        )
        q_min = (
            torch.iinfo(out_qconf.dtype).min
            if out_qconf.quant_min is None
            else out_qconf.quant_min
        )

        scale = 1 / (q_max - q_min + 1)

        output_obs_ctr = observer = FixedQParamsObserver.with_args(
            scale=scale,
            zero_point=0,
            dtype=quantization_config.output_activation.dtype,
            qscheme=torch.torch.per_tensor_affine,
            quant_max=q_max,
            quant_min=q_min,
        )
        if quantization_config in (
            get_8a8w_qnn_qat_config(),
            get_16a4w_qnn_qat_config(),
        ):
            output_obs_ctr = FixedQParamsFakeQuantize.with_args(
                observer=observer,
                scale=scale,
                zero_point=0,
                dtype=quantization_config.output_activation.dtype,
                qscheme=torch.torch.per_tensor_affine,
                quant_max=q_max,
                quant_min=q_min,
            )

        # make sigmoid map to the range between 0~1
        out_act_quantization_spec = QuantizationSpec(
            dtype=quantization_config.output_activation.dtype,
            quant_max=q_max,
            quant_min=q_min,
            observer_or_fake_quant_ctr=output_obs_ctr,
            qscheme=torch.torch.per_tensor_affine,
        )

        if _is_float_tensor(node):
            node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=out_act_quantization_spec,
                _annotated=True,
            )


@register_annotator(
    [torch.ops.aten.hardswish.default, torch.ops.aten.hardswish_.default],
    QnnConstants.OpHardSwish.op_name,
)
class HardSwish(GeneralOpDef):
    pass


# TODO: The index_copy op cannot directly map to OpScatterNd because the index input in torch is not a tensor.
@register_annotator(
    [torch.ops.aten.index_copy.default, torch.ops.aten.index_copy_.default], qnn_op=None
)
class IndexCopy(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        # Avoid annotating the input node because mutable buffers will be folded during the convert_pt2e process.
        value = node.args[3]

        input_qspec_map = {}
        input_qspec_map[value] = quantization_config.input_activation

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=SharedQuantizationSpec((value, node)),
            _annotated=True,
        )


# TODO: The index_put op cannot directly map to OpScatterNd because the index input in torch is not a tensor.
@register_annotator(
    [torch.ops.aten.index_put.default, torch.ops.aten.index_put_.default], qnn_op=None
)
class IndexPut(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        # Avoid annotating the input node because mutable buffers will be folded during the convert_pt2e process.
        value = node.args[2]

        input_qspec_map = {}
        input_qspec_map[value] = quantization_config.input_activation

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=SharedQuantizationSpec((value, node)),
            _annotated=True,
        )


@register_annotator(
    [torch.ops.aten.layer_norm.default], QnnConstants.OpLayerNorm.op_name
)
class LayerNorm(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        act_node = node.args[0]
        weight_node = node.args[2]
        bias_node = None
        if len(node.args) > 2:
            bias_node = node.args[3]

        if _is_annotated([node]):
            return
        input_act_qspec = quantization_config.input_activation

        annotate_input_qspec_map(
            node,
            act_node,
            input_act_qspec,
        )
        if input_act_qspec.dtype == torch.int32:
            annotate_input_qspec_map(
                node,
                weight_node,
                get_16a16w_qnn_ptq_config().weight,
            )
        else:
            annotate_input_qspec_map(
                node,
                weight_node,
                input_act_qspec,
            )
        nodes_to_mark_annotated = [node, weight_node]
        if bias_node:
            annotate_input_qspec_map(
                node,
                bias_node,
                quantization_config.bias,
            )
            nodes_to_mark_annotated.append(bias_node)
        annotate_output_qspec(node, quantization_config.output_activation)
        _mark_nodes_as_annotated(nodes_to_mark_annotated)

    @staticmethod
    def validate(
        node: Node, constraints_list: List[NormalizedConstraints], soc_info: SocInfo
    ) -> bool:
        valid = True
        if not _is_annotated([node]):
            return valid

        weight_node = node.args[1]
        weight_qspec = node.meta[Q_ANNOTATION_KEY].input_qspec_map.get(
            weight_node, None
        )

        act_node = node.args[0]
        act_qspec = node.meta[Q_ANNOTATION_KEY].input_qspec_map.get(act_node, None)
        if (
            act_qspec
            and act_qspec.dtype == torch.int32
            and weight_qspec
            and weight_qspec.dtype == torch.int32
        ):
            valid &= validate_16a16w_support(soc_info)
            if not valid:
                logging.warning(
                    f"16-bit activations + 16-bit weights requires V73 or newer for {node.name}"
                )

        valid &= validate_against_backend_constraints(node, constraints_list)
        return valid


@register_annotator([torch.ops.aten.lt.Tensor], QnnConstants.OpElementWiseLess.op_name)
class Less(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)


@register_annotator(
    [torch.ops.aten.le.Tensor], QnnConstants.OpElementWiseLessEqual.op_name
)
class LessEqual(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.linalg_vector_norm.default], qnn_op=None)
class LinalgVectorNorm(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.linear.default], QnnConstants.OpFullyConnected.op_name
)
class Linear(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        if _is_annotated([node]):
            return

        # block quantization
        if quantization_config.block_size is not None:
            quantization_config.weight.observer_or_fake_quant_ctr.p.keywords.update(
                {QCOM_BLOCK_SIZE: quantization_config.block_size}
            )

        input_qspec_map = {}
        act_node = node.args[0]
        assert isinstance(act_node, Node)
        input_spec = quantization_config.input_activation
        input_qspec_map[act_node] = input_spec

        weight_node = node.args[1]
        assert isinstance(weight_node, Node)
        input_qspec_map[weight_node] = quantization_config.weight

        if len(node.args) > 2:
            bias_node = node.args[2]
            if isinstance(bias_node, Node):
                if callable(quantization_config.bias):
                    input_qspec_map[bias_node] = quantization_config.bias(node)
                else:
                    input_qspec_map[bias_node] = quantization_config.bias

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=quantization_config.output_activation,
            _annotated=True,
        )

        # We use get_source_partition in pass, but it is the same source for MultiheadAttention, so we need to change its source_fn_stack.
        node.meta["source_fn_stack"] = [(node, torch.nn.Linear)]

    @staticmethod
    def validate(
        node: Node, constraints_list: List[NormalizedConstraints], soc_info: SocInfo
    ) -> bool:
        valid = True
        if not _is_annotated([node]):
            return valid

        weight_node = node.args[1]
        weight_qspec = node.meta[Q_ANNOTATION_KEY].input_qspec_map.get(
            weight_node, None
        )
        if (
            weight_qspec
            and weight_qspec.observer_or_fake_quant_ctr.p.keywords.get(
                QCOM_BLOCK_SIZE, None
            )
            is not None
        ):
            valid &= validate_lpbq_support(soc_info)
            if not valid:
                logging.warning(
                    f"LPBQ (16a4w block-wise quantization) requires V69 or newer for {node.name}"
                )

        act_node = node.args[0]
        act_qspec = node.meta[Q_ANNOTATION_KEY].input_qspec_map.get(act_node, None)
        if (
            act_qspec
            and act_qspec.dtype == torch.int32
            and weight_qspec
            and weight_qspec.dtype == torch.int32
        ):
            valid &= validate_16a16w_support(soc_info)
            if not valid:
                logging.warning(
                    f"16-bit activations + 16-bit weights requires V73 or newer for {node.name}"
                )
        valid &= validate_against_backend_constraints(node, constraints_list)
        return valid


@register_annotator([torch.ops.aten.log.default], QnnConstants.OpElementWiseLog.op_name)
class Log(GeneralOpDef):
    pass


@register_annotator([torch.ops.aten.log_softmax.int], QnnConstants.OpLogSoftmax.op_name)
class LogSoftmax(GeneralOpDef):
    pass


# TODO: The masked_fill op cannot directly map to OpElementWiseSelect because the number of inputs is different
#       from what is expected by OpElementWiseSelect.
@register_annotator([torch.ops.aten.masked_fill.Tensor], qnn_op=None)
class MaskedFill(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        if _is_annotated([node]):
            return

        input_qspec_map = {}
        for input_node in node.args:
            assert isinstance(input_node, Node)
            if _is_float_tensor(input_node):
                input_qspec_map[input_node] = quantization_config.input_activation

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=(
                quantization_config.output_activation
                if _is_float_tensor(node)
                else None
            ),
            _annotated=True,
        )


@register_annotator(
    [torch.ops.aten.bmm.default, torch.ops.aten.matmul.default],
    QnnConstants.OpMatMul.op_name,
)
class MatMul(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        if _is_annotated([node]):
            return

        input_act_qspec = quantization_config.input_activation
        output_act_qspec = quantization_config.output_activation

        input_qspec_map = {}
        input_act0 = node.args[0]
        if isinstance(input_act0, Node):
            input_qspec_map[input_act0] = input_act_qspec

        input_act1 = node.args[1]
        if isinstance(input_act1, Node):
            # In matmul, QNN_DATATYPE_SFIXED_POINT_16 Input1 must have QNN_DATATYPE_UFIXED_POINT_16 Input0 and must be symmetric quantized.
            if input_act_qspec.dtype == torch.int32:
                # we should use int16 for mm / bmm instead of int4
                input_qspec_map[input_act1] = get_16a16w_qnn_ptq_config().weight
            else:
                input_qspec_map[input_act1] = input_act_qspec

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=output_act_qspec,
            _annotated=True,
        )

    @staticmethod
    def validate(
        node: Node, constraints_list: List[NormalizedConstraints], soc_info: SocInfo
    ) -> bool:
        valid = True
        if not _is_annotated([node]):
            return valid
        weight_node = node.args[1]
        weight_qspec = node.meta[Q_ANNOTATION_KEY].input_qspec_map.get(
            weight_node, None
        )

        act_node = node.args[0]
        act_qspec = node.meta[Q_ANNOTATION_KEY].input_qspec_map.get(act_node, None)
        if (
            act_qspec
            and act_qspec.dtype == torch.int32
            and weight_qspec
            and weight_qspec.dtype == torch.int16
        ):
            valid &= validate_16a16w_support(soc_info)
            if not valid:
                logging.warning(
                    f"16-bit activations + 16-bit weights requires V73 or newer for {node.name}"
                )
        valid &= validate_against_backend_constraints(node, constraints_list)
        return valid


@register_annotator(
    [torch.ops.aten.max.other, torch.ops.aten.maximum.default],
    QnnConstants.OpElementWiseMaximum.op_name,
)
class Max(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)


@register_annotator(
    [
        torch.ops.aten.max_pool2d.default,
        torch.ops.aten.max_pool2d_with_indices.default,
        torch.ops.aten.adaptive_max_pool2d.default,
    ],
    QnnConstants.OpPoolMax2d.op_name,
)
class MaxPool2d(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.min.other, torch.ops.aten.minimum.default],
    QnnConstants.OpElementWiseMinimum.op_name,
)
class Min(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)


@register_annotator(
    [torch.ops.aten.mul, torch.ops.aten.mul.Tensor, torch.ops.aten.mul_.Tensor],
    QnnConstants.OpElementWiseMultiply.op_name,
)
class Mul(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.neg.default], QnnConstants.OpElementWiseNeg.op_name)
class Neg(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.ne.Tensor], QnnConstants.OpElementWiseNotEqual.op_name
)
class NotEqual(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)


@register_annotator(
    [torch.ops.aten.bitwise_or.Tensor, torch.ops.aten.__or__.Tensor],
    QnnConstants.OpElementWiseOr.op_name,
)
class Or(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.pad.default], QnnConstants.OpPad.op_name)
class Pad(GeneralOpDef):
    pass


@register_annotator(
    [
        torch.ops.aten.permute.default,
        torch.ops.aten.swapaxes.default,
        torch.ops.aten.transpose.int,
    ],
    QnnConstants.OpTranspose.op_name,
)
class Permute(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_in_out_obs_sharing_op(node, quantization_config)
        if not _is_annotated([node]):
            annotate_single_in_share_out(node, quantization_config)


@register_annotator(
    [torch.ops.aten.pixel_shuffle.default], QnnConstants.OpDepthToSpace.op_name
)
class PixelShuffle(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.pixel_unshuffle.default], QnnConstants.OpSpaceToDepth.op_name
)
class PixelUnshuffle(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.pow.Tensor_Tensor, torch.ops.aten.square.default],
    QnnConstants.OpElementWisePower.op_name,
)
class Pow(GeneralOpDef):
    pass


@register_annotator([torch.ops.aten.prelu.default], QnnConstants.OpPRelu.op_name)
class PReLU(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.max.dim], QnnConstants.OpReduceMax.op_name)
class ReduceMax(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_single_in(node, quantization_config)


@register_annotator(
    [torch.ops.aten.mean.default, torch.ops.aten.mean.dim],
    QnnConstants.OpReduceMean.op_name,
)
class ReduceMean(GeneralOpDef):
    pass


@register_annotator([torch.ops.aten.min.dim], QnnConstants.OpReduceMin.op_name)
class ReduceMin(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_single_in(node, quantization_config)


@register_annotator([torch.ops.aten.sum.dim_IntList], QnnConstants.OpReduceSum.op_name)
class ReduceSum(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)


@register_annotator(
    [torch.ops.aten.relu.default, torch.ops.aten.relu_.default],
    QnnConstants.OpRelu.op_name,
)
class Relu(GeneralOpDef):
    pass


@register_annotator(
    [
        torch.ops.aten.clamp.default,
        torch.ops.aten.clamp_min.default,
        torch.ops.aten.clamp_max.default,
        torch.ops.aten.hardtanh.default,
        torch.ops.aten.hardtanh_.default,
    ],
    QnnConstants.OpReluMinMax.op_name,
)
class ReluMinMax(GeneralOpDef):
    pass


@register_annotator(
    [
        torch.ops.aten.expand.default,
        torch.ops.aten.repeat.default,
    ],
    QnnConstants.OpTile.op_name,
)
class Repeat(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_in_out_obs_sharing_op(node, quantization_config)
        if not _is_annotated([node]):
            annotate_single_in_share_out(node, quantization_config)


# TODO: Expand_as op cannot directly map to QNN OpTile due to the number of input doesn't match.
@register_annotator(
    [
        torch.ops.aten.expand_as.default,
    ],
    qnn_op=None,
)
class ExpandAs(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_in_out_obs_sharing_op(node, quantization_config)
        if not _is_annotated([node]):
            annotate_single_in_share_out(node, quantization_config)


@register_annotator(
    [
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.reshape.default,
        torch.ops.aten.squeeze.default,
        torch.ops.aten.squeeze.dim,
        torch.ops.aten.squeeze_copy.dims,
        torch.ops.aten.unflatten.int,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.unsqueeze.default,
        torch.ops.aten.unsqueeze_copy.default,
        torch.ops.aten.view_copy.default,
        torch.ops.aten.view.default,
    ],
    QnnConstants.OpReshape.op_name,
)
class Reshape(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_in_out_obs_sharing_op(node, quantization_config)
        if not _is_annotated([node]):
            annotate_single_in_share_out(node, quantization_config)


@register_annotator([torch.ops.aten.rms_norm.default], QnnConstants.OpRmsNorm.op_name)
class RmsNorm(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        if _is_annotated([node]):
            return

        act_node = node.args[0]

        # TODO current only support 16a16w
        annotate_input_qspec_map(
            node,
            act_node,
            quantization_config.input_activation,
        )

        if len(node.args) > 2 and node.args[2] is not None:
            weight_node = node.args[2]
            annotate_input_qspec_map(
                node,
                weight_node,
                quantization_config.input_activation,
            )
        nodes_to_mark_annotated = [node]
        annotate_output_qspec(node, quantization_config.output_activation)
        _mark_nodes_as_annotated(nodes_to_mark_annotated)


# TODO: There is a bug in the BackendOpInfo library, so it is bypassed now.
@register_annotator([torch.ops.aten.rsqrt.default], qnn_op=None)
class Rsqrt(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.round.default], QnnConstants.OpElementWiseRound.op_name
)
class Round(GeneralOpDef):
    pass


@register_annotator([torch.ops.aten.scaled_dot_product_attention.default], qnn_op=None)
class ScaledDotProductAttention(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.sigmoid, torch.ops.aten.sigmoid.default],
    QnnConstants.OpSigmoid.op_name,
)
class Sigmoid(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        if _is_annotated([node]):
            return

        input_qspec_map = {}
        input_act = node.args[0]
        input_qspec_map[input_act] = quantization_config.input_activation

        assert isinstance(input_act, Node)
        out_qconf = quantization_config.output_activation

        q_max = (
            torch.iinfo(out_qconf.dtype).max
            if out_qconf.quant_max is None
            else out_qconf.quant_max
        )
        q_min = (
            torch.iinfo(out_qconf.dtype).min
            if out_qconf.quant_min is None
            else out_qconf.quant_min
        )

        scale = 1 / (q_max - q_min + 1)

        output_obs_ctr = observer = FixedQParamsObserver.with_args(
            scale=scale,
            zero_point=0,
            dtype=quantization_config.output_activation.dtype,
            qscheme=torch.torch.per_tensor_affine,
            quant_max=q_max,
            quant_min=q_min,
        )
        if quantization_config in (
            get_8a8w_qnn_qat_config(),
            get_16a4w_qnn_qat_config(),
        ):
            output_obs_ctr = FixedQParamsFakeQuantize.with_args(
                observer=observer,
                scale=scale,
                zero_point=0,
                dtype=quantization_config.output_activation.dtype,
                qscheme=torch.torch.per_tensor_affine,
                quant_max=q_max,
                quant_min=q_min,
            )

        # make sigmoid map to the range between 0~1
        out_act_quantization_spec = QuantizationSpec(
            dtype=quantization_config.output_activation.dtype,
            quant_max=q_max,
            quant_min=q_min,
            observer_or_fake_quant_ctr=output_obs_ctr,
            qscheme=torch.torch.per_tensor_affine,
        )

        if _is_float_tensor(node):
            node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=out_act_quantization_spec,
                _annotated=True,
            )


@register_annotator(
    [torch.ops.aten.sign.default], QnnConstants.OpElementWiseSign.op_name
)
class Sign(GeneralOpDef):
    pass


@register_annotator([torch.ops.aten.sin.default], QnnConstants.OpElementWiseSin.op_name)
class Sin(GeneralOpDef):
    pass


# TODO: The slice_scatter op cannot directly map to QNN OpScatterNd due to the order of input doesn't match.
@register_annotator([torch.ops.aten.slice_scatter.default], qnn_op=None)
class SliceScatter(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        if _is_annotated([node]):
            return

        input = node.args[0]
        value = node.args[1]

        input_qspec_map = {}
        input_qspec_map[input] = quantization_config.input_activation
        input_qspec_map[value] = SharedQuantizationSpec((input, node))

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=SharedQuantizationSpec((input, node)),
            _annotated=True,
        )


@register_annotator(
    [
        torch.ops.aten.softmax.int,
        torch.ops.aten._softmax.default,
        torch.ops.aten._safe_softmax.default,
    ],
    QnnConstants.OpSoftmax.op_name,
)
class Softmax(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.sqrt.default], QnnConstants.OpElementWiseSquareRoot.op_name
)
class Sqrt(GeneralOpDef):
    pass


@register_annotator([torch.ops.aten.stack.default], QnnConstants.OpPack.op_name)
class Stack(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        if _is_annotated([node]) or not _is_float_tensor(node):
            return

        input_nodes = node.args[0]
        assert isinstance(input_nodes, Sequence)

        first_input_node = input_nodes[0]
        input_qspec_map = {}
        assert isinstance(first_input_node, Node)
        input_qspec_map[first_input_node] = quantization_config.input_activation
        share_qparams_with_input_act0_qspec = SharedQuantizationSpec(
            (first_input_node, node)
        )

        for input_node in input_nodes[1:]:
            if input_node not in input_qspec_map:
                assert isinstance(input_node, Node)
                input_qspec_map[input_node] = share_qparams_with_input_act0_qspec

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=share_qparams_with_input_act0_qspec,
            _annotated=True,
        )


@register_annotator(
    [
        torch.ops.aten.flip.default,
        torch.ops.aten.narrow.default,
        torch.ops.aten.select.int,
        torch.ops.aten.slice.Tensor,
    ],
    QnnConstants.OpStridedSlice.op_name,
)
class StrideSlice(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.sub, torch.ops.aten.sub.Tensor, torch.ops.aten.rsub.Tensor],
    QnnConstants.OpElementWiseSubtract.op_name,
)
class Sub(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)


@register_annotator([torch.ops.aten.tanh.default], QnnConstants.OpTanh.op_name)
class Tanh(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        if _is_annotated([node]):
            return

        input_qspec_map = {}
        input_act = node.args[0]
        assert isinstance(input_act, Node)
        input_qspec_map[input_act] = quantization_config.input_activation

        out_act_quantization_spec = quantization_config.output_activation
        # Based on quantization constraints in QNN document, for the uint16 data type, the scale should be set to 1/32768.0 and the zero_point should be 32768.
        if out_act_quantization_spec.dtype == torch.int32:
            scale = 1 / 32768.0
            zero_point = 32768
            output_obs_ctr = observer = FixedQParamsObserver.with_args(
                scale=scale,
                zero_point=zero_point,
                dtype=quantization_config.output_activation.dtype,
                qscheme=torch.torch.per_tensor_affine,
                quant_max=quantization_config.output_activation.quant_max,
                quant_min=quantization_config.output_activation.quant_min,
            )
            if isinstance(
                quantization_config.output_activation.observer_or_fake_quant_ctr,
                torch.ao.quantization.fake_quantize.FakeQuantizeBase,
            ):
                output_obs_ctr = FixedQParamsFakeQuantize.with_args(
                    observer=observer,
                    scale=scale,
                    zero_point=zero_point,
                    dtype=quantization_config.output_activation.dtype,
                    qscheme=torch.torch.per_tensor_affine,
                    quant_max=quantization_config.output_activation.quant_max,
                    quant_min=quantization_config.output_activation.quant_min,
                )

            out_act_quantization_spec = QuantizationSpec(
                dtype=quantization_config.output_activation.dtype,
                quant_max=quantization_config.output_activation.quant_max,
                quant_min=quantization_config.output_activation.quant_min,
                observer_or_fake_quant_ctr=output_obs_ctr,
                qscheme=torch.torch.per_tensor_affine,
            )

        if _is_float_tensor(node):
            node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                input_qspec_map=input_qspec_map,
                output_qspec=out_act_quantization_spec,
                _annotated=True,
            )


@register_annotator([torch.ops.aten.topk.default], QnnConstants.OpTopK.op_name)
class Topk(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        if _is_annotated([node]):
            return

        input_qspec_map = {}
        if _is_float_tensor(node.args[0]):
            input_act = node.args[0]
            assert isinstance(input_act, Node)
            input_qspec_map[input_act] = quantization_config.input_activation

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=SharedQuantizationSpec((input_act, node)),
            _annotated=True,
        )


@register_annotator(
    [
        torch.ops.aten.conv_transpose1d.default,
        torch.ops.aten.conv_transpose2d.input,
    ],
    QnnConstants.OpTransposeConv2d.op_name,
)
class TransposeConv2d(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_conv(node, quantization_config)

    @staticmethod
    def validate(
        node: Node, constraints_list: List[NormalizedConstraints], soc_info: SocInfo
    ) -> bool:
        valid = True
        if not _is_annotated([node]):
            return valid

        weight_node = node.args[1]
        weight_qspec = node.meta[Q_ANNOTATION_KEY].input_qspec_map.get(
            weight_node, None
        )

        act_node = node.args[0]
        act_qspec = node.meta[Q_ANNOTATION_KEY].input_qspec_map.get(act_node, None)
        if (
            act_qspec
            and act_qspec.dtype == torch.int32
            and weight_qspec
            and weight_qspec.dtype == torch.int32
        ):
            valid &= validate_16a16w_support(soc_info)
            if not valid:
                logging.warning(
                    f"16-bit activations + 16-bit weights requires V73 or newer for {node.name}"
                )

        valid &= validate_against_backend_constraints(node, constraints_list)
        return valid


@register_annotator(
    [
        torch.ops.aten.conv_transpose3d.input,
    ],
    QnnConstants.OpTransposeConv3d.op_name,
)
class TransposeConv3d(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_conv(node, quantization_config)


@register_annotator([torch.ops.aten.unbind.int], QnnConstants.OpUnpack.op_name)
class Ubind(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        input_act = node.args[0]
        # Seems like unbind.int can be either float or int. Only quant when input is float.
        if _is_annotated([node]) or not _is_float_tensor(input_act):
            return
        input_qspec_map = {}

        assert isinstance(input_act, Node)
        share_qparams_with_out_node0_qspec = SharedQuantizationSpec((input_act, node))
        input_qspec_map[input_act] = quantization_config.input_activation

        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=share_qparams_with_out_node0_qspec,
            _annotated=True,
        )

        for user in node.users:
            user.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
                output_qspec=share_qparams_with_out_node0_qspec,
                _annotated=True,
            )


@register_annotator(
    [torch.ops.aten.upsample_bicubic2d.vec], QnnConstants.OpResize.op_name
)
class UpsampleBicubic2d(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.upsample_bilinear2d.vec], QnnConstants.OpResizeBilinear.op_name
)
class UpsampleBilinear2d(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.upsample_nearest2d.vec],
    QnnConstants.OpResizeNearestNeighbor.op_name,
)
class UpsampleNearest2d(GeneralOpDef):
    pass


@register_annotator(
    [torch.ops.aten.where.self, torch.ops.aten.where.ScalarSelf],
    QnnConstants.OpElementWiseSelect.op_name,
)
class Where(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        if _is_annotated([node]):
            return

        input_qspec_map = {}
        for input_node in node.args:
            assert isinstance(input_node, Node)
            if _is_float_tensor(input_node):
                input_qspec_map[input_node] = quantization_config.input_activation
        node.meta[Q_ANNOTATION_KEY] = QuantizationAnnotation(
            input_qspec_map=input_qspec_map,
            output_qspec=(
                quantization_config.output_activation
                if _is_float_tensor(node)
                else None
            ),
            _annotated=True,
        )


@register_annotator(
    [torch.ops.aten.bitwise_xor.Tensor, torch.ops.aten.__xor__.Tensor],
    QnnConstants.OpElementWiseXor.op_name,
)
class Xor(GeneralOpDef):
    @staticmethod
    def annotate(node: Node, quantization_config: QuantizationConfig) -> None:
        annotate_binary(node, quantization_config)
