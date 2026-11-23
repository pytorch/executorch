# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Build ``GraphBundle``s for a model whose quantization stage was skipped.

Used by compilation for FP16 and GPU flows, so a component-keyed model still
reaches the per-component fan-out.
"""

from typing import Any, Dict, Optional

from executorch.backends.qualcomm.genai_pipeline.export_inputs import to_export_inputs
from executorch.backends.qualcomm.genai_pipeline.graph_bundle import GraphBundle
from executorch.backends.qualcomm.genai_pipeline.graph_roles import is_deployed


def build_unquantized_bundles(
    model_module: Dict[str, Any],
    example_inputs: Dict[str, Dict[str, Any]],
    meta: Optional[Dict[str, Dict[str, Dict[str, Any]]]],
) -> Dict[str, Dict[str, GraphBundle]]:
    """Bundle each deployable graph, unquantized.

    Quantize-only graphs are dropped: without quantization they have no use.

    Args:
        model_module: The prepared modules, keyed by component.
        example_inputs: Model-preparation inputs, keyed by component and graph.
        meta: Graph metadata, keyed by component and graph.

    Returns:
        Unquantized bundles, keyed by component and graph name.

    Raises:
        ValueError: If a graph has no role definition, or a component has no
            module.
    """
    meta = meta or {}

    graphs: Dict[str, Dict[str, GraphBundle]] = {}
    for component, component_inputs in example_inputs.items():
        if component not in model_module:
            raise ValueError(f"Component '{component}' has no prepared module")

        graphs[component] = {
            graph_name: GraphBundle(
                module=model_module[component],
                inputs=to_export_inputs(component, graph_name, inputs, meta),
                meta=meta.get(component, {}).get(graph_name, {}),
                quantized=False,
            )
            for graph_name, inputs in component_inputs.items()
            if is_deployed(component, graph_name)
        }

    return graphs
