# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Sequence

import torch


def materialize_symints(
    graph: torch.fx.Graph, values: Sequence[int | torch.SymInt]
) -> list[torch.fx.Node | int]:
    """Materialize symbolic integers into FX graph values.

    Args:
        graph (torch.fx.Graph): Graph where producer nodes are inserted.
        values (Sequence[int | torch.SymInt]): Integer values to materialize.

    Returns:
        list[torch.fx.Node | int]: Materialized graph nodes or static integers.

    """
    materialized = graph.materialize_symints(values)
    if not any(isinstance(value, torch.SymInt) for value in materialized):
        return materialized
    raise AssertionError("materialize_symints returned a raw SymInt")
