# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Turn model-preparation example inputs into a graph's export signature.

Shared by quantization, which exports the graphs to calibrate them, and by
compilation, which exports them directly when quantization is skipped.
"""

from typing import Any, Dict

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_TEXT_DECODER,
)

META_USE_KV_CACHE = "get_use_kv_cache"


def to_export_inputs(
    component: str,
    graph_name: str,
    example_inputs: Any,
    meta: Dict[str, Dict[str, Dict[str, Any]]],
) -> Any:
    """Flatten text-decoder inputs into their positional export signature.

    Args:
        component: Artifact component key for the graph.
        graph_name: Graph key used to retrieve metadata.
        example_inputs: Model-preparation inputs in structured decoder form.
        meta: Component- and graph-keyed model metadata.

    Returns:
        Unchanged inputs for non-decoder components. For text decoder graphs,
        positional inputs with attention masks expanded and position IDs/KV
        caches included only when ``get_use_kv_cache`` is enabled.

    Raises:
        ValueError: If a text decoder graph lacks ``get_use_kv_cache``.
    """
    if component != ARTIFACT_TEXT_DECODER:
        return example_inputs

    graph_meta = meta.get(component, {}).get(graph_name, {})
    if META_USE_KV_CACHE not in graph_meta:
        raise ValueError(
            f"Component '{component}' graph '{graph_name}' is missing "
            f"{META_USE_KV_CACHE} metadata"
        )

    use_kv_cache = graph_meta[META_USE_KV_CACHE]
    return (
        example_inputs[0],
        *example_inputs[1],
        *((example_inputs[2],) if use_kv_cache else []),
        *(example_inputs[3] if use_kv_cache else []),
        *(example_inputs[4] if use_kv_cache else []),
    )
