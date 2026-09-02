# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-component lowering settings for a GenAI model.

The graphs of one model do not all want the same lowering settings: only the
decoder asks for MHA-to-SHA conversion and multiple contexts, encoders do not
share weights, and encoders do not take their IO from RPC memory. This maps each
artifact key to the settings that component needs, so the compilation strategy
has one table to consult rather than a chain of ``if`` branches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, TYPE_CHECKING

from executorch.backends.qualcomm.genai_pipeline.artifact_keys import (
    ARTIFACT_ATTENTION_SINK_EVICTOR,
    ARTIFACT_AUDIO_ENCODER,
    ARTIFACT_TEXT_DECODER,
    ARTIFACT_TEXT_ENCODER,
    ARTIFACT_TOK_EMBEDDING,
    ARTIFACT_VISION_ENCODER,
)

if TYPE_CHECKING:
    from executorch.backends.qualcomm.genai_pipeline.compilation.compile_spec_builder import (
        QnnCompileSpecBuilder,
    )
    from executorch.backends.qualcomm.genai_pipeline.control_args import ControlArgs
    from executorch.exir.backend.compile_spec_schema import CompileSpec

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComponentSpecPolicy:
    """How one component's compile specs differ from the defaults.

    Attributes:
        use_mha2sha: Convert multi-head to single-head attention. Decoder only.
        allow_multi_contexts: Whether this component honours sharding. Only the
            decoder is sharded, so only it can ask for multiple contexts.
        allow_weight_sharing: Whether the component's graphs may share weights.
            Off for encoders, which are lowered one graph per ``.pte``, so there
            is nothing to share and requesting it only constrains the compiler.
        allow_online_prepare: Whether ``online_prepare`` applies. Encoders are
            compiled ahead of time regardless.
        uses_shared_buffer: Whether the runner supplies the graph IO from RPC
            memory, so memory planning must not allocate it. Off for encoders.
    """

    use_mha2sha: bool = False
    allow_multi_contexts: bool = False
    allow_weight_sharing: bool = True
    allow_online_prepare: bool = True
    uses_shared_buffer: bool = True


# Mirrors the three ``generate_htp_compiler_spec`` call sites in the legacy
# ``llama.py`` and the ``ExecutorchBackendConfig`` of each legacy lowering.
# Components absent from this table use the default policy.
_POLICIES: Dict[str, ComponentSpecPolicy] = {
    ARTIFACT_TEXT_DECODER: ComponentSpecPolicy(
        use_mha2sha=True,
        allow_multi_contexts=True,
    ),
    ARTIFACT_TOK_EMBEDDING: ComponentSpecPolicy(),
    ARTIFACT_ATTENTION_SINK_EVICTOR: ComponentSpecPolicy(),
    ARTIFACT_AUDIO_ENCODER: ComponentSpecPolicy(
        allow_weight_sharing=False,
        allow_online_prepare=False,
        uses_shared_buffer=False,
    ),
    ARTIFACT_TEXT_ENCODER: ComponentSpecPolicy(
        allow_weight_sharing=False,
        allow_online_prepare=False,
        uses_shared_buffer=False,
    ),
    ARTIFACT_VISION_ENCODER: ComponentSpecPolicy(
        allow_weight_sharing=False,
        allow_online_prepare=False,
        uses_shared_buffer=False,
    ),
}

_DEFAULT_POLICY = ComponentSpecPolicy()


def get_component_policy(artifact_key: str) -> ComponentSpecPolicy:
    """Look up a component's spec policy.

    Args:
        artifact_key: The component's artifact key (see ``artifact_keys``).

    Returns:
        The component's policy, or the default for an unrecognised key.
    """
    policy = _POLICIES.get(artifact_key)
    if policy is None:
        logger.debug(
            "No spec policy for artifact '%s'; using the defaults", artifact_key
        )
        return _DEFAULT_POLICY

    return policy


def build_component_compile_specs(
    builder: "QnnCompileSpecBuilder",
    artifact_key: str,
    graph_names: List[str],
    control_args: "ControlArgs",
    num_sharding: int = 1,
    quantized: bool = True,
) -> Dict[str, List["CompileSpec"]]:
    """Build one component's compile specs, one entry per graph.

    Every graph of a component gets the same settings -- the legacy flow builds
    one spec and replicates it across the component's graph names -- but each
    graph needs its own list, since lowering keys them by graph name.

    ``shared_buffer`` is deliberately left to the builder's default
    (``not enable_x86_64``) rather than taken from ``control_args.shared_buffer``:
    the latter is a *runner* flag, and every legacy compile-spec site uses the
    former. Passing the runner flag here would disable shared buffers on device.

    Args:
        builder: The builder for this component's target.
        artifact_key: The component being built for (see ``artifact_keys``).
        graph_names: The component's graph names.
        control_args: The flow's control arguments.
        num_sharding: This component's resolved shard count. Requests multiple
            contexts when greater than one, and only for a component that
            honours sharding.
        quantized: Whether the component's graphs were quantized. An
            unquantized component is compiled in FP16, as in the legacy flow.

    Returns:
        The component's compile specs, keyed by graph name.
    """
    policy = get_component_policy(artifact_key)

    specs = builder.build(
        use_fp16=not quantized,
        use_multi_contexts=policy.allow_multi_contexts and num_sharding > 1,
        use_weight_sharing=None if policy.allow_weight_sharing else False,
        online_prepare=policy.allow_online_prepare and control_args.online_prepare,
        use_mha2sha=policy.use_mha2sha,
    )

    return {graph_name: specs for graph_name in graph_names}


def build_executorch_config(artifact_key: str) -> Any:
    """Build one component's ``to_executorch`` configuration.

    A shared-buffer component leaves its graph IO unallocated, since the runner
    passes RPC memory for it. ``BuildQuantIo`` always runs, so quantized IO
    tensors get their types.

    Args:
        artifact_key: The component being built for (see ``artifact_keys``).

    Returns:
        An ``ExecutorchBackendConfig`` for the component.
    """
    from executorch.backends.qualcomm._passes.build_quant_io import BuildQuantIo
    from executorch.exir.capture._config import ExecutorchBackendConfig
    from executorch.exir.passes.memory_planning_pass import MemoryPlanningPass

    if not get_component_policy(artifact_key).uses_shared_buffer:
        return ExecutorchBackendConfig(passes=[BuildQuantIo()])

    return ExecutorchBackendConfig(
        memory_planning_pass=MemoryPlanningPass(
            alloc_graph_input=False,
            alloc_graph_output=False,
        ),
        passes=[BuildQuantIo()],
    )
