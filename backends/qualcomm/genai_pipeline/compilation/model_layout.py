# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Model-specific module paths and tagging rules used during lowering.

Sharding splits a graph on module paths, and the decoder's quantized-IO tagger
has a few model-specific rules. Model preparation supplies the layout (see
``ModelPreparationOutputConfig.model_layout``), so compilation receives it as
data and does no lookup by model name. Values that the graph's ``meta`` already
carries, such as a second head dim or a layer offset, are read from ``meta``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class ModelLayout:
    """Where a model's layers live, and which extra tensors its decoder tags.

    Attributes:
        decoder_layer_prefixes: Module-path prefixes of the decoder's layers.
            The sharding pattern matches these.
        encoder_layer_prefixes: Module-path prefixes of an encoder's layers.
        offset_layer_prefix: For a decoder whose layers span two modules that
            each count from zero, the prefix whose indices are offset by
            ``meta["get_n_self_layers"]``. ``None`` for one layer sequence.
        per_layer_input_traces: Source snippets. A node whose stack trace holds
            one is tagged with ``io_type``.
        donor_kv_prefixes: Source-name prefixes of the donor K/V tensors that
            shared layers consume. A matching node is tagged with ``io_type``.

    .. note::
        The tagging rules match variable names in the model's ``forward``.
        Renaming them there silently drops the tag.
    """

    decoder_layer_prefixes: Tuple[str, ...] = ("layers",)
    encoder_layer_prefixes: Tuple[str, ...] = (
        "encoder.layers",
        "vision_tower.encoder.layer",
        "vision_model.encoder.layers",
    )
    offset_layer_prefix: Optional[str] = None
    per_layer_input_traces: Tuple[str, ...] = ()
    donor_kv_prefixes: Tuple[str, ...] = ()
