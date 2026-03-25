# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.import copy

import torch

from executorch.backends.cortex_m.quantizer.quantizer import CortexMQuantizer
from executorch.backends.nxp import NXP_NEUTRON_BACKEND_IGNORE
from executorch.backends.nxp.backend.neutron_target_spec import NeutronTargetSpec
from executorch.backends.nxp.quantizer.neutron_quantizer import NeutronQuantizer
from torch.fx import GraphModule, Node
from torchao.quantization.pt2e.quantizer import Quantizer
from torchao.quantization.pt2e.quantizer.quantizer import Q_ANNOTATION_KEY


class IMXRT700CMQuantizer(Quantizer):
    """Hybrid quantizer that uses NeutronQuantizer for Neutron supported ops and CortexMQuantizer for the rest."""

    def __init__(
        self,
        target_spec: NeutronTargetSpec,
        cortex_m_designated_node_identifiers: set[str],
    ):
        """
        :param target_spec: Neutron target specification
        :param cortex_m_designated_node_identifiers: Set of identifiers (stored in node.meta["torch_fn"][0]) of nodes
                                                      that are not supported by Neutron. Nodes not in this set will use
                                                      the NeutronQuantizer. Nodes in this set and nodes skipped by
                                                      NeutronQuantizer will be quantized with the CortexMQuantizer.
        """
        super().__init__()
        self.neutron_quantizer = NeutronQuantizer(target_spec)

        def _should_quantize_with_cortex_m(n: Node) -> bool:
            # The CortexMQuantizer should be used for nodes which were explicitly chosen, and for nodes which were not
            #  quantized by the NeutronQuantizer.
            return (
                n.meta.get(NXP_NEUTRON_BACKEND_IGNORE, False)
                or Q_ANNOTATION_KEY not in n.meta
            )

        self.cortex_m_quantizer = CortexMQuantizer(
            filter_fn=_should_quantize_with_cortex_m
        )

        self.target_spec = target_spec
        self.cortex_m_designated_node_identifiers = cortex_m_designated_node_identifiers

    def annotate(self, model: GraphModule) -> GraphModule:
        # Due to how SharedQuantizationSpecs are used, skipping select ops in NeutronQuantizer would cause errors.
        # Instead, first NeutronQuantizer is used to annotate all nodes in the model (that it supports). Then, the
        #  annotations are removed from nodes that should be handled by the Cortex-M backend, and the CortexMQuantizer
        #  is applied to these nodes only.

        # Apply NeutronQuantizer.
        model = self.neutron_quantizer.annotate(model)

        # Mark the nodes which will be handled by the Cortex-M backend.
        # This mark is used by the CortexMQuantizer, and later by the NeutronPartitioner.
        self._mark_nodes_to_be_quantized_by_cortex_m_quantizer(
            model, self.cortex_m_designated_node_identifiers
        )

        # Remove the annotations from nodes selected for the Cortex-M quantizer.
        for node in model.graph.nodes:
            if node.op != "call_function":
                continue
            if not node.meta.get(NXP_NEUTRON_BACKEND_IGNORE, False):
                continue  # Node should be quantized by NeutronQuantizer. All good.

            # This node should be quantized using CortexMQuantizer. Remove the quantization annotation.
            node.meta.pop(Q_ANNOTATION_KEY, None)
            node.meta.pop("quantizer_matched", None)

        # Apply CortexMQuantizer.
        model = self.cortex_m_quantizer.annotate(model)

        return model

    def transform_for_annotation(
        self, model: torch.fx.GraphModule
    ) -> torch.fx.GraphModule:
        model = self.neutron_quantizer.transform_for_annotation(model)
        model = self.cortex_m_quantizer.transform_for_annotation(model)
        return model

    def validate(self, model: GraphModule) -> None:
        self.neutron_quantizer.validate(model)
        self.cortex_m_quantizer.validate(model)

    # noinspection PyMethodMayBeStatic
    def _mark_nodes_to_be_quantized_by_cortex_m_quantizer(
        self, graph: GraphModule, cortex_m_designated_node_identifiers: set[str]
    ):
        """Mark nodes which were selected to be handled by to Cortex-M backend.
        The mark is `node.meta[NXP_NEUTRON_BACKEND_IGNORE] = True`
        """
        for node in graph.graph.nodes:
            if (torch_fn := node.meta.get("torch_fn")) is not None and torch_fn[
                0
            ] in cortex_m_designated_node_identifiers:
                # This node was selected specifically for the Cortex-M backend.
                node.meta[NXP_NEUTRON_BACKEND_IGNORE] = True
