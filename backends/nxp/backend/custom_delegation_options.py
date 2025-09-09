# Copyright 2025 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from dataclasses import dataclass


@dataclass
class CustomDelegationOptions:
    """The class allows the user to specify details which affect which nodes will be delegated."""

    # Neutron requires the channel dimension to be multiple of `num_macs` for concatenation (cat op).
    #  Due to different dim ordering in torch (channel_first) and Neutron IR (channel last), dim of the channel is
    #  ambiguous. Cat converter will defensively require both possible dimension index for the channels to be multiple
    #  of `num_macs`. The `force_delegate_cat` allows the user to turn off the defensive check if from the model design
    #  it is known this constraint will be satisfied.
    force_delegate_cat: bool = False
