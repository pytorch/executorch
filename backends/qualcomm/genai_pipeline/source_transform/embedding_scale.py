# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State-dict transforms for the token embedding table."""

from __future__ import annotations

from typing import Any, Dict


def scale_token_embedding(
    state_dict: Dict[str, Any], *, embedding_scale_factor: float
) -> Dict[str, Any]:
    """Bake ``embedding_scale_factor`` into the embedding table.

    A no-op -- dtype included -- for models whose factor is 1.0, and for models
    that export their token embedding as a separate graph and so carry no such
    key. Declared only by the rows whose params file sets a factor; the guard
    means a row that declares it anyway costs nothing.
    """
    key = "tok_embeddings.weight"
    if embedding_scale_factor != 1.0 and key in state_dict:
        state_dict[key] = state_dict[key].float() * embedding_scale_factor
    return state_dict
