# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Keys identifying the compiled artifacts a GenAI model produces.

``CompilationOutputConfig.artifact_paths`` is keyed by these names, one entry
per ``.pte`` file. The values match the on-device runner's ``pte_paths`` keys
exactly, so a compiled bundle can be handed to the runner without translation.

.. note::
    These are **artifact** keys, not **graph** names. One artifact may hold
    several methods: a hybrid text decoder lowers its AR-N prefill and AR-1
    decode graphs into a single multi-method ``.pte`` so they can share
    weights, and that one file is reached through
    ``ARTIFACT_TEXT_DECODER``. Graph names (``kv_forward`` /
    ``prefill_forward``) live in a separate key space used *inside* a lowering
    call.

    The same string constants exist in the legacy
    ``examples/qualcomm/oss_scripts/llama/decoder_constants.py``. They are
    duplicated here rather than imported so that this package does not depend
    on the example scripts; ``tests/test_artifact_keys.py`` asserts the two
    stay in agreement, and the legacy copy goes away with the legacy flow.
"""

from __future__ import annotations

# Text decoder: the language model itself. Always present.
ARTIFACT_TEXT_DECODER = "text_decoder"

# Token embedding, lowered separately for multimodal models so that the
# embedding lookup can run while modality features are being inserted.
ARTIFACT_TOK_EMBEDDING = "tok_embedding"

# Attention sink evictor, compiled only when the attention sink feature is on.
ARTIFACT_ATTENTION_SINK_EVICTOR = "attention_sink_evictor"

# Modality encoders. Present only for the corresponding multimodal model.
ARTIFACT_AUDIO_ENCODER = "audio_encoder"
ARTIFACT_TEXT_ENCODER = "text_encoder"
ARTIFACT_VISION_ENCODER = "vision_encoder"

# Every key this package may emit.
#
# Membership in ``artifact_paths`` is meaningful: the runner decides whether a
# model is multimodal by testing for the encoder keys, so an artifact that was
# not compiled must be **absent** rather than mapped to ``None``.
ALL_ARTIFACT_KEYS = frozenset(
    {
        ARTIFACT_ATTENTION_SINK_EVICTOR,
        ARTIFACT_AUDIO_ENCODER,
        ARTIFACT_TEXT_DECODER,
        ARTIFACT_TEXT_ENCODER,
        ARTIFACT_TOK_EMBEDDING,
        ARTIFACT_VISION_ENCODER,
    }
)

# The decode graph's QDQ exported program, written by quantization before the
# calibration graph is released and read back by the SQNR evaluation.
#
# A **filename**, not an artifact key: it is a ``.pt2`` exported program rather
# than a lowered ``.pte``, the runner never sees it, and only the A-side stages
# touch it. Hence deliberately absent from ``ALL_ARTIFACT_KEYS``. It lives here
# because this module already owns the names the pipeline writes to disk.
DECODE_QDQ_FILENAME = "decode_qdq.pt2"
