# Copyright 2025-2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass


@dataclass
class CustomDelegationOptions:
    """The class allows the user to specify details which affect which nodes will be delegated."""

    # Proposed partitions which only contain Neutron no-ops are normally not delegated, as the NeutronConverter would
    #  not create any NeutronGraph that can be called. This is done by the partitioner itself, and is not handled by
    #  the individual node converters.
    allow_no_op_partitions: bool = False
