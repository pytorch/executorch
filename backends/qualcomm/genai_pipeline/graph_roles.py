# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Which graphs of a component are calibrated, deployed, or both.

Shared by quantization, which calibrates and propagates encodings, and by
compilation, which lowers only deployable graphs when quantization is skipped.
"""

from enum import auto, Enum
from typing import Dict

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_AUDIO_ENCODER,
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_TEXT_ENCODER,
    ARTIFACT_TOK_EMBEDDING,
    ARTIFACT_VISION_ENCODER,
)
from executorch.backends.qualcomm.genai_pipeline.graph_names import (
    DECODER_GRAPH_NAMES,
    GRAPH_FORWARD,
    TOK_EMBEDDING_GRAPH_NAMES,
)


class GraphRole(Enum):
    """The role of one graph in quantization and deployment."""

    # Uses real calibration data but is never lowered for deployment.
    QUANTIZE = auto()
    # Initializes encodings, receives overrides from a quantize graph, then lowers.
    DEPLOY = auto()
    # Uses calibration data and lowers as the same graph.
    SHARED = auto()


GRAPH_ROLES: Dict[str, Dict[str, GraphRole]] = {
    ARTIFACT_TEXT_DECODER: {
        GRAPH_FORWARD: GraphRole.QUANTIZE,
        **dict.fromkeys(DECODER_GRAPH_NAMES, GraphRole.DEPLOY),
    },
    ARTIFACT_TOK_EMBEDDING: {
        GRAPH_FORWARD: GraphRole.QUANTIZE,
        **dict.fromkeys(TOK_EMBEDDING_GRAPH_NAMES, GraphRole.DEPLOY),
    },
    ARTIFACT_AUDIO_ENCODER: {GRAPH_FORWARD: GraphRole.SHARED},
    ARTIFACT_TEXT_ENCODER: {GRAPH_FORWARD: GraphRole.SHARED},
    ARTIFACT_VISION_ENCODER: {GRAPH_FORWARD: GraphRole.SHARED},
}


def get_graph_role(component: str, graph_name: str) -> GraphRole:
    """Look up one graph's role.

    Args:
        component: Artifact component key.
        graph_name: Graph key within the component.

    Returns:
        The graph's role.

    Raises:
        ValueError: If the component or graph has no role definition.
    """
    if (component_roles := GRAPH_ROLES.get(component)) is None:
        raise ValueError(f"Component '{component}' has no graph-role definition")
    if (role := component_roles.get(graph_name)) is None:
        raise ValueError(
            f"Component '{component}' has no role for graph '{graph_name}'"
        )
    return role


def is_deployed(component: str, graph_name: str) -> bool:
    """Whether a graph is lowered for deployment.

    Args:
        component: Artifact component key.
        graph_name: Graph key within the component.

    Returns:
        ``False`` only for a quantize-only graph.
    """
    return get_graph_role(component, graph_name) is not GraphRole.QUANTIZE
