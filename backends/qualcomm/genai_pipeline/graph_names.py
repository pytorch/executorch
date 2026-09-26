# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Names identifying the graphs (methods) inside one compiled artifact.

A hybrid decoder is exported twice from the same weights -- an AR-1 decode graph
and an AR-N prefill graph -- and both become methods of a single multi-method
``.pte`` so they share weights. These are the method names.

.. note::
    These are **graph** names, not **artifact** keys (see ``artifact_keys``).
    Graph names are used *inside* one lowering call; artifact keys name the
    ``.pte`` files that call produces.

    The deployed graph names also exist in the legacy
    ``examples/qualcomm/oss_scripts/llama/decoder_constants.py``. They are
    duplicated here rather than imported so that this package does not depend on
    the example scripts; ``tests/test_graph_names.py`` asserts the two stay in
    agreement, and the legacy copy goes away with the legacy flow.
"""

from __future__ import annotations

# Decode: AR-1, consumes and updates the KV cache one token at a time.
GRAPH_KV_FORWARD = "kv_forward"

# Prefill: AR-N, processes the prompt in one pass.
GRAPH_PREFILL_FORWARD = "prefill_forward"

# The decoder's graphs, decode first.
#
# Order is significant: in ``kv`` mode there is no prefill graph, and the legacy
# flow slices this list to the number of graphs a model actually built. Decode is
# also the authoritative graph for the artifact's constant methods, since the
# runtime uses its KV-cache scales for both graphs.
DECODER_GRAPH_NAMES = [GRAPH_KV_FORWARD, GRAPH_PREFILL_FORWARD]

# The token-embedding graphs, in the same order for the same reason.
GRAPH_TOK_EMBEDDING_KV_FORWARD = "tok_embedding_kv_forward"
GRAPH_TOK_EMBEDDING_PREFILL_FORWARD = "tok_embedding_prefill_forward"

TOK_EMBEDDING_GRAPH_NAMES = [
    GRAPH_TOK_EMBEDDING_KV_FORWARD,
    GRAPH_TOK_EMBEDDING_PREFILL_FORWARD,
]

# The name ExecuTorch gives a module exported as a single graph.
#
# Used for the components that build one graph rather than a decode/prefill pair
# -- an encoder -- and for the decoder's full-auto-regressive calibration graph,
# which sources encodings and is released before compilation. It is therefore
# absent from ``DECODER_GRAPH_NAMES``: those are the *deployed* graphs.
GRAPH_FORWARD = "forward"
