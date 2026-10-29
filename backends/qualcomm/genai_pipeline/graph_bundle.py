# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The per-graph unit handed from quantization to compilation.

One decoder is exported several times from the same weights -- an AR-N prefill
graph and an AR-1 decode graph differ only in the ``ar_len`` baked into them --
so the stages after model preparation operate on a *set* of graphs rather than a
single module. ``GraphBundle`` is that set's element: everything compilation
needs about one graph, in one object, so the pipeline threads a single
``{graph_name: GraphBundle}`` map instead of several parallel
``{graph_name: value}`` dicts that can fall out of step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

#: The supported key sets ``GraphBundle.quant_io_dtypes`` carries when it is
#: not ``None``. Graphs without KV caches only carry ``io_type``.
_QUANT_IO_DTYPE_KEY_SETS = (
    frozenset({"kv_type", "io_type"}),
    frozenset({"io_type"}),
)


@dataclass(frozen=True)
class GraphBundle:
    """One deployable graph and the inputs needed to lower it.

    Frozen so that a bundle cannot be edited in place as it moves between
    stages: a stage that needs to change one field returns a new bundle via
    ``dataclasses.replace``, which keeps the producer's output readable after
    the consumer has run.

    .. note::
        Immutability stops at the field boundary. ``module`` is a
        ``GraphModule`` and ``meta`` a dict, and both are mutated by the
        quantization stage -- ``convert_pt2e`` writes the quantized logits and
        KV-cache attributes into ``meta`` in place. ``frozen=True`` protects
        which objects a bundle names, not their contents.

    .. note::
        Only **deployable** graphs are bundled. A hybrid decoder also builds a
        full-auto-regressive calibration graph, but that exists solely to source
        activation statistics: quantization propagates its scales onto the
        deployed graphs and then releases it, so it never reaches compilation.

    ``compile_spec``, ``dep_table`` and ``passes_job`` are deliberately not
    fields: they are settings of the *lowering call* rather than of a graph, so
    the compilation stage derives them from ``meta``, ``ControlArgs`` and the
    shard count. ``skip_node_id_set`` / ``skip_node_op_set`` are per-run
    partitioning overrides and travel in ``extra_options`` until two graphs of
    one model need different values.

    Attributes:
        module: The graph to lower, quantized and already reconciled against the
            calibration graph. A ``torch.fx.GraphModule`` in practice.
        inputs: Positional example inputs for ``torch.export``, derived from the
            **model** and never from the calibration data: they carry this
            graph's positional signature, its zero-initialized KV caches, and
            the AR length baked in because HTP has no dynamic shapes.
        meta: The graph's ``get_metadata()`` constants -- layer count, head dim,
            context and AR lengths -- plus the quantized logits / KV-cache
            attributes written during ``convert_pt2e``. Becomes the ``.pte``'s
            constant methods, so it is complete only **after** quantization has
            run.
        quant_io_dtypes: The graph-boundary dtypes quantization chose.
            Compilation builds the ``TagQuantIO`` pass settings from these; it
            cannot derive them, since they come from the quantization recipe's
            KV and logits bit widths.

            A non-``None`` mapping has one of these shapes:

            * ``{"kv_type": torch.dtype, "io_type": torch.dtype}`` for a
              graph with a KV cache, such as a decoder.
            * ``{"io_type": torch.dtype}`` for a graph without a KV cache,
              such as an encoder. Encoders derive ``io_type`` from their
              quantization recipe, but have no ``kv_type`` to record.

            ``None`` means *this graph's boundary dtypes were not chosen by a
            recipe* -- which happens for two different reasons, and compilation
            must not treat them alike:

            * **Quantization was skipped** for the graph, so its IO stays
              float32 and ``TagQuantIO`` is left inactive.
            * **No fixed-point boundary dtype was selected.** For example, a
              recipe bit width may not map to a fixed-point dtype. The whole
              mapping is then ``None`` rather than a mapping with one missing
              key.

            ``io_type`` is required whenever the mapping is present.
            ``kv_type`` is required only for graphs with a KV cache.
            ``__post_init__`` rejects every other key combination.
        modality_inputs: Encoder inputs for a multimodal model, keyed by
            modality. ``None`` for a text-only model.
        executorch_config: Optional override for the ``to_executorch``
            configuration. A property of the *lowering call* rather than of one
            graph, so a multi-method ``.pte`` uses one config for the whole
            group; carried here so a caller can vary it without reaching into
            ``extra_options``.
    """

    module: Any
    inputs: Tuple[Any, ...]
    meta: Dict[str, Any] = field(default_factory=dict)
    quant_io_dtypes: Optional[Dict[str, Any]] = None
    modality_inputs: Optional[Dict[str, Any]] = None
    executorch_config: Optional[Any] = None

    def __post_init__(self) -> None:
        """Reject an unsupported ``quant_io_dtypes`` key combination.

        Raises:
            ValueError: If ``quant_io_dtypes`` is a mapping whose keys are not
                ``io_type`` alone or ``kv_type`` and ``io_type`` together.
        """
        if self.quant_io_dtypes is None:
            return

        keys = frozenset(self.quant_io_dtypes)
        if keys not in _QUANT_IO_DTYPE_KEY_SETS:
            valid_key_sets = " or ".join(
                str(sorted(key_set)) for key_set in _QUANT_IO_DTYPE_KEY_SETS
            )
            raise ValueError(
                "quant_io_dtypes must carry exactly "
                f"{valid_key_sets} or be None; got {sorted(keys)}"
            )
