# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Gate finite-score static TopK lowering and its index interfaces."""

import torch.fx
from executorch.backends.arm._passes.decompose_topk_pass import (
    get_static_topk_config,
    topk_indices_only_feed_int32_casts,
    TOPK_OPS,
)
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa.specification import TosaSpecification


@register_tosa_support_check
class TopKSupported(SupportedTOSAOperatorCheck):
    """Accept the static decomposition under its finite-score precondition."""

    tosa_specs = TosaSpecification.all_versions_for_profile("FP")
    targets = list(TOPK_OPS)

    def is_node_tosa_supported(
        self, node: torch.fx.Node, tosa_spec: TosaSpecification
    ) -> bool:
        """Check the supported metadata and prepared index boundary."""
        config, reason = get_static_topk_config(node, tosa_spec)
        if config is None:
            self.reporter.report_reject(node, f"Unsupported TopK: {reason}")
            return False
        if not topk_indices_only_feed_int32_casts(node):
            self.reporter.report_reject(
                node, "TopK index users require an int32 narrowing boundary."
            )
            return False
        return True
