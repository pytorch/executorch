# Copyright (c) Intel Corporation
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch.fx
from torch.ao.quantization.observer import HistogramObserver
from torch.ao.quantization.observer import PerChannelMinMaxObserver
from torch.ao.quantization.quantizer.quantizer import EdgeOrNode
from torch.ao.quantization.quantizer.quantizer import QuantizationAnnotation
from torch.ao.quantization.quantizer.quantizer import QuantizationSpec
from torch.ao.quantization.quantizer.quantizer import QuantizationSpecBase
from torch.ao.quantization.quantizer.quantizer import Quantizer
from torch.ao.quantization.quantizer.quantizer import SharedQuantizationSpec

import nncf
import nncf.common.quantization as q
import nncf.experimental.torch.fx as nncf_fx
import nncf.parameters as p
import nncf.quantization.advanced_parameters as advanced_p
from nncf.common.graph.graph import NNCFGraph

QUANT_ANNOTATION_KEY = "quantization_annotation"


class OpenVINOQuantizer(Quantizer):
    """
    Implementation of the Torch AO quantizer which annotates models with quantization annotations
    optimally for the inference via OpenVINO.
    """

    def __init__(
        self,
        *,
        mode: Optional[p.QuantizationMode] = None,
        preset: Optional[q.structs.QuantizationPreset] = None,
        target_device: p.TargetDevice = p.TargetDevice.ANY,
        transformer_model: bool = False,
        ignored_scope: Optional[nncf.IgnoredScope] = None,
        overflow_fix: Optional[advanced_p.OverflowFix] = None,
        quantize_outputs: bool = False,
        activations_quantization_params: Optional[advanced_p.QuantizationParameters] = None,
        weights_quantization_params: Optional[advanced_p.QuantizationParameters] = None,
    ):
        """
        :param mode: Defines optimization mode for the algorithm. None by default.
        :param preset: A preset controls the quantization mode (symmetric and asymmetric).
            It can take the following values:
            - `performance`: Symmetric quantization of weights and activations.
            - `mixed`: Symmetric quantization of weights and asymmetric quantization of activations.
            Default value is None. In this case, `mixed` preset is used for `transformer`
            model type otherwise `performance`.
        :param target_device: A target device the specificity of which will be taken
            into account while compressing in order to obtain the best performance
            for this type of device, defaults to TargetDevice.ANY.
        :param model_type: Model type is needed to specify additional patterns
            in the model. Supported only `transformer` now.
        :param ignored_scope: An ignored scope that defined the list of model control
            flow graph nodes to be ignored during quantization.
        :param overflow_fix: This option controls whether to apply the overflow issue
            fix for the 8-bit quantization.
        :param quantize_outputs: Whether to insert additional quantizers right before
            each of the model outputs.
        :param activations_quantization_params: Quantization parameters for model
            activations.
        :param weights_quantization_params: Quantization parameters for model weights.
        """
        self._min_max_algo = nncf.quantization.algorithms.min_max.algorithm.MinMaxQuantization(
            mode=mode,
            preset=preset,
            target_device=target_device,
            model_type=p.ModelType.TRANSFORMER if transformer_model else None,
            ignored_scope=ignored_scope,
            overflow_fix=overflow_fix,
            quantize_outputs=quantize_outputs,
            activations_quantization_params=activations_quantization_params,
            weights_quantization_params=weights_quantization_params,
        )

    def get_nncf_quantization_setup(
        self, model: torch.fx.GraphModule, nncf_graph: NNCFGraph
    ) -> q.quantizer_setup.SingleConfigQuantizerSetup:
        self._min_max_algo._set_backend_entity(model)
        return self._min_max_algo.find_quantization_setup(model, nncf_graph)

    def annotate(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        nncf_graph = nncf_fx.nncf_graph_builder.GraphConverter.create_nncf_graph(model)
        quantization_setup = self.get_nncf_quantization_setup(model, nncf_graph)

        graph = model.graph
        node_vs_torch_annotation = defaultdict(QuantizationAnnotation)

        for qp in quantization_setup.quantization_points.values():
            edge_or_node, annotation = self._get_edge_or_node_and_annotation(
                graph, nncf_graph, qp, node_vs_torch_annotation
            )
            qspec = self._get_torch_ao_qspec_from_qp(qp)
            self._fill_torch_ao_annotation(edge_or_node, qspec, annotation)

        for quantizer_ids in quantization_setup.unified_scale_groups.values():

            root_quantizer_id = self._get_unified_scales_root_quantizer_id(
                nncf_graph, quantizer_ids, quantization_setup
            )
            root_qp = quantization_setup.quantization_points[root_quantizer_id]

            if any(root_qp.qconfig != quantization_setup.quantization_points[q_id].qconfig for q_id in quantizer_ids):
                qps = [quantization_setup.quantization_points[q_id] for q_id in quantizer_ids]
                msg = (
                    "Different quantization configs are set to one unified scale group:"
                    f"{[(qp.insertion_point.__dict__, str(qp.qconfig)) for qp in qps]}"
                )
                raise nncf.InternalError(msg)

            root_target_node = nncf_fx.node_utils.get_graph_node_by_name(
                graph, root_qp.insertion_point.target_node_name
            )
            root_edge_or_node = self._get_edge_or_node(root_target_node, root_qp, nncf_graph)

            for quantizer_id in quantizer_ids:
                if quantizer_id == root_quantizer_id:
                    continue

                qspec = SharedQuantizationSpec(root_edge_or_node)
                qp = quantization_setup.quantization_points[quantizer_id]
                edge_or_node, annotation = self._get_edge_or_node_and_annotation(
                    graph, nncf_graph, qp, node_vs_torch_annotation
                )
                self._fill_torch_ao_annotation(edge_or_node, qspec, annotation)

        for node, annotation in node_vs_torch_annotation.items():
            assert QUANT_ANNOTATION_KEY not in node.meta
            node.meta[QUANT_ANNOTATION_KEY] = annotation
        return model

    @staticmethod
    def _get_unified_scales_root_quantizer_id(
        nncf_graph: NNCFGraph, quantizer_ids: List[int], quantizer_setup: q.quantizer_setup.SingleConfigQuantizerSetup
    ) -> int:
        """
        Identifies the earliest quantizer node ID based on the corresponding `nncf_node.node_id`
        in the given NNCFGraph. This is required by the `_get_obs_or_fq_map` function.
        Refer to: https://github.com/pytorch/pytorch/blob/main/torch/ao/quantization/pt2e/prepare.py#L291

        :param nncf_graph: The NNCFGraph instance.
        :param quantizer_ids: The list of quantizer IDs to evaluate.
        :param quantizer_setup: The instance of SingleConfigQuantizerSetup.
        :return: The ID of the earliest quantizer node in terms of `nncf_node.node_id`.
        """
        nncf_node_quantizer_id = None
        root_quantizer_id = None
        for quantizer_id in quantizer_ids:
            target_node_name = quantizer_setup.quantization_points[quantizer_id].insertion_point.target_node_name
            nncf_node = nncf_graph.get_node_by_name(target_node_name)
            if nncf_node_quantizer_id is None or nncf_node.node_id < nncf_node_quantizer_id:
                root_quantizer_id = quantizer_id
                nncf_node_quantizer_id = nncf_node.node_id
        return root_quantizer_id

    @staticmethod
    def _get_edge_or_node_and_annotation(
        graph: torch.fx.Graph,
        nncf_graph: NNCFGraph,
        qp: q.quantizer_setup.QuantizationPointBase,
        node_vs_torch_annotation: Dict[torch.fx.Node, QuantizationAnnotation],
    ) -> Tuple[EdgeOrNode, QuantizationAnnotation]:
        """
        Retrieves the edge or node and its corresponding QuantizationAnnotation based on the given graph,
        quantization point, and node-to-annotation mapping.

        :param graph: torch.fx.Graph instance.
        :param nncf_graph: NNCFGraph instance.
        :param qp: QuantizationPointBase instance.
        :param node_vs_torch_annotation: A dictionary mapping torch.fx.GraphNode objects to their respective
            QuantizationAnnotations.
        :return: A tuple containing the EdgeOrNode and its associated QuantizationAnnotation.
        """
        target_node = nncf_fx.node_utils.get_graph_node_by_name(graph, qp.insertion_point.target_node_name)
        annotation = node_vs_torch_annotation[target_node]
        edge_or_node = OpenVINOQuantizer._get_edge_or_node(target_node, qp, nncf_graph)
        return edge_or_node, annotation

    @staticmethod
    def _get_edge_or_node(
        target_node: torch.fx.Node, qp: q.quantizer_setup.QuantizationPointBase, nncf_graph: NNCFGraph
    ) -> EdgeOrNode:
        """
        Returns the edge or node based on the given target node and quantization point.

        :param target_node: Target node instance.
        :param qp: QuantizationPointBase instance.
        :param graph: NNCFGraph instance.
        :return: The corresponding EdgeOrNode derived from the target node and quantization point.
        """
        ip = qp.insertion_point
        if qp.is_weight_quantization_point():
            nncf_node = nncf_graph.get_node_by_name(target_node.name)
            weights_ports_ids = nncf.torch.model_graph_manager.get_weight_tensor_port_ids(nncf_node, nncf_graph)
            if len(weights_ports_ids) > 1:
                # TODO(dlyakhov): support quantization for nodes with several weights
                nncf.common.logging.nncf_logger.warning(
                    f"Quantization of the weighted node {target_node.name}"
                    " is not yet supported by the OpenVINOQuantizer."
                    f" Only the weight on port ID {weights_ports_ids[0]} will be quantized."
                    f" Quantizable weights are located on ports: {weights_ports_ids}."
                )
            weight_node = target_node.all_input_nodes[weights_ports_ids[0]]
            return (weight_node, target_node)

        if ip.input_port_id is None:
            return target_node

        node = target_node.all_input_nodes[ip.input_port_id]
        return (node, target_node)

    @staticmethod
    def _fill_torch_ao_annotation(
        edge_or_node: EdgeOrNode,
        qspec: QuantizationSpecBase,
        annotation_to_update: QuantizationAnnotation,
    ) -> None:
        """
        Helper method to update the annotation_to_update based on the specified edge_or_node and qspec.

        :param edge_or_node: The target EdgeOrNode to be used for the update.
        :param qspec: An instance of QuantizationSpecBase representing the quantization specification to apply.
        :param annotation_to_update: The annotation to update based on the edge_or_node and qspec.
        """
        if isinstance(edge_or_node, torch.fx.Node):
            annotation_to_update.output_qspec = qspec
        else:
            annotation_to_update.input_qspec_map[edge_or_node[0]] = qspec

    @staticmethod
    def _get_torch_ao_qspec_from_qp(qp: q.quantizer_setup.QuantizationPointBase) -> QuantizationSpec:
        """
        Retrieves the quantization configuration from the given quantization point and
        converts it into a QuantizationSpec.

        :param qp: An instance of QuantizationPointBase.
        :return: A QuantizationSpec retrieved and converted from the quantization point.
        """
        # Eps value is copied from nncf/torch/quantization/layers.py
        extra_args = {"eps": 1e-16}
        qconfig = qp.qconfig
        is_weight = qp.is_weight_quantization_point()

        if qconfig.per_channel:
            torch_qscheme = (
                torch.per_channel_symmetric
                if qconfig.mode is q.structs.QuantizationScheme.SYMMETRIC
                else torch.per_channel_affine
            )
        else:
            torch_qscheme = (
                torch.per_tensor_symmetric
                if qconfig.mode is q.structs.QuantizationScheme.SYMMETRIC
                else torch.per_tensor_affine
            )
        if is_weight:
            observer = PerChannelMinMaxObserver
            quant_min = -128
            quant_max = 127
            dtype = torch.int8
            channel_axis = 0
        else:
            observer = (
                HistogramObserver
                if torch_qscheme in [torch.per_tensor_symmetric, torch.per_tensor_affine]
                else PerChannelMinMaxObserver
            )
            quant_min = 0
            quant_max = 255
            dtype = torch.int8 if qconfig.signedness_to_force else torch.uint8
            channel_axis = 1  # channel dim for activations
        return QuantizationSpec(
            dtype=dtype,
            observer_or_fake_quant_ctr=observer.with_args(**extra_args),
            quant_min=quant_min,
            quant_max=quant_max,
            qscheme=torch_qscheme,
            ch_axis=channel_axis,
            is_dynamic=False,
        )

    def validate(self, model: torch.fx.GraphModule) -> None:
        pass

    def transform_for_annotation(self, model: torch.fx.GraphModule) -> torch.fx.GraphModule:
        nncf_fx.transformations.fold_constant_except_qdq(model)
        return model
