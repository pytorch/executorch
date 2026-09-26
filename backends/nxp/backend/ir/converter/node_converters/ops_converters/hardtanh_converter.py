# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from executorch.backends.nxp.backend.ir.converter.node_converters.ops_converters.clamp_converter import (
    ClampConverter,
)
from torch.fx import Node


class HardTanhConverter(ClampConverter):
    @staticmethod
    def _get_bounds(node: Node) -> tuple[float | None, float | None]:
        args = node.args

        match len(args):
            case 1:
                min_val = -1
                max_val = 1

            case 2:
                min_val = args[1]
                max_val = 1

            case 3:
                min_val = args[1]
                max_val = args[2]

            case _:
                # should not occur
                raise ValueError(
                    f"Unexpected number of arguments for HardTanh node: {len(args)}"
                )

        return min_val, max_val
