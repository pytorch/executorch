# Copyright 2025-2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Provide TOSA support checks for ``aten.index.Tensor``.

Reject unsupported patterns such as high-rank index tensors, front-positioned
slice/ellipsis/None markers, and cases that exceed ``int32`` element limits.

"""

import math

import torch
import torch.fx as fx
from executorch.backends.arm._passes.arm_pass_utils import get_first_fake_tensor
from executorch.backends.arm.common.type import ensure_type
from executorch.backends.arm.operator_support.tosa_supported_operators import (
    register_tosa_support_check,
    SupportedTOSAOperatorCheck,
)
from executorch.backends.arm.tosa import TosaSpecification
from executorch.exir.dialects._ops import ops as exir_ops


@register_tosa_support_check
class IndexTensorSupported(SupportedTOSAOperatorCheck):
    """Prevent partitioning of unsupported ``index.Tensor`` usages.

    This support check is intended to prevent the partitioning of
    currently unsupported usages of the index.Tensor operator.

    1. Usages where indexing tensors are of rank 4 or higher.
        This is due to the AnnotateChannelsLastDimOrder pass and
        the rarity of such operation.
        Support is possible but would require further changes to the above
        pass which can be added at such a time as is necessary.

    2. Usages where slice, ellipsis or None are present before an indexing tensor:
        t[{start}:{end}, indexTensor] - slicing
        t[None, indexTensor] - unsqueeze
        t[..., indexTensor] - ellipsis

    3. Usages where the value tensor contains more than int32.max elements
        This is due to int32 TOSA limitation and the fact that we flatten out
        and accumulate all index tensors.
        As such to avoid overflow we reject lowering of this operator if it is
        possible for indices to go over the int32 limit.

    Extra information regarding #2:
        Pytorch decomposes slice and None usages before they reach aten.
        In the case of Slicing and Unsqueeze, Pytorch will add the relevant
        operation just before the index.Tensor op.
        In the case of Ellipsis no extra operation is added.

        In all three cases Pytorch will insert "None"(s) in the index list
        only if the above operations are done on a dimension BEFORE one being indexed.

        When slicing, unsqueeze and ellipsis are done on dimensions after
        the ones being indexed, then they do not affect the final output
        values, only the shape. Thus None is not passed to the index.Tensor op.

        The purpose of None is to signify to index.Tensor that a dimension
        should not be indexed.
        In such cases the logic behaves similar to batching along that dimension.
        For the sake of simplicity we have not implemented this behavior yet
        and thus have put this support check in place to prevent the partitioning
        of index.Tensor ops which include None.

    Examples:
        #1 - Slice -----------------------------------------------------
        t = torch.randint(25, size(25, 3, 6))
        t[1:5, torch.arange(3)]

        Turns into: (edge pseudo code)
        slice_res = ...edge__ops_aten_slice_copy_Tensor(t, dim=0, start=1, end=2)
        out = ...edge__ops_aten_index_Tensor(slice_res, [None, torch.arange(3)])

        #2 - None (Unsqueeze) ------------------------------------------
        t = torch.randint(25, size(25, 3, 6))
        t[None, torch.arange(3)]

        Turns into: edge pseudo code)
        unsqueeze_res = ...edge__ops_aten_unsqueeze(t, dim=0)
        out = ...edge__ops_aten_index_Tensor(unsqueeze_res, [None, torch.arange(3)])

        #3 - None (Unsqueeze) After index ------------------------------
        t = torch.randint(25, size(25, 3, 6))
        t[torch.arange(3), None]

        Turns into: edge pseudo code)
        unsqueeze_res = ...edge__ops_aten_unsqueeze(t, dim=1)
        out = ...edge__ops_aten_index_Tensor(unsqueeze_res, [torch.arange(3)])

    NB.
        With the current implementation of flattening tensors and indices out,
        supporting None (Unsqueeze) is simply a matter of ignoring the
        None dimension.
        This is not the case for Slice and Ellipsis operators, where
        the size of the new dimension can be > 1.

        Note that slice ops interleaved between indexes such as:
            t[1:3, torch.arange(5), 2:3, torch.arange(3).reshape(3,1)]
        are also possible and can result in some unintuitive behaviors
        where batching and indexing are mixed together.

    """

    targets = [exir_ops.edge.aten.index.Tensor]

    tosa_specs = [
        TosaSpecification.create_from_string("TOSA-1.0+INT"),
        TosaSpecification.create_from_string("TOSA-1.0+FP"),
        TosaSpecification.create_from_string("TOSA-1.1+INT"),
        TosaSpecification.create_from_string("TOSA-1.1+FP"),
    ]

    def is_node_tosa_supported(
        self, node: fx.Node, tosa_spec: TosaSpecification
    ) -> bool:  # type: ignore[override, misc]
        """Return True if ``aten.index.Tensor`` usage fits supported patterns.

        Enforces the following constraints:
        - No ``None`` (unsqueeze), slice, or ellipsis before an indexing tensor.
        - Indexing tensors have rank <= 3.
        - The value tensor element count fits in ``int32``.

        """
        indices = node.args[1]
        for index in indices:  # type: ignore[union-attr]
            # Usage 2 guard
            if index is None:
                self.reporter.report_reject(
                    node,
                    (
                        "None (from slice/unsqueeze/ellipsis) before an indexing tensor"
                        " is not supported."
                    ),
                )
                return False

            # Usage 1 guard
            index = ensure_type(torch.fx.Node, index)
            fake_tensor = get_first_fake_tensor(index)
            if len(fake_tensor.size()) > 3:
                self.reporter.report_reject(
                    node,
                    ("Indexing tensors of rank >= 4 is not supported."),
                )
                return False

        # Usage 3 guard
        input_node = ensure_type(torch.fx.Node, node.args[0])
        total_vals = math.prod(get_first_fake_tensor(input_node).shape)
        if total_vals > torch.iinfo(torch.int32).max:
            self.reporter.report_reject(
                node,
                ("Value size exceeds int32 range; would overflow flattened indexing."),
            )
            return False

        return True
