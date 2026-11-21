# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import executorch.backends.arm.tosa.dialect  # noqa: F401
import torch
from executorch.backends.arm.tosa.specification import (
    TosaLoweringContext,
    TosaSpecification,
)
from executorch.backends.arm.vgf._passes import RewriteCustomSDPAToTosaCustomPass
from executorch.backends.arm.vgf.tosa_custom_sdpa import (
    CUSTOM_SDPA_DOMAIN_NAME,
    custom_sdpa_operator_name,
    decode_payload,
    ensure_custom_sdpa_registered,
)
from executorch.exir import to_edge
from executorch.exir.dialects._ops import ops as exir_ops
from torch.export import export


class _DecodeCustomSDPA(torch.nn.Module):
    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        q_bshd = q.transpose(1, 2).contiguous()
        k_bshd = k.transpose(1, 2).contiguous()
        v_bshd = v.transpose(1, 2).contiguous()
        output = torch.ops.llama.custom_sdpa.default(
            q_bshd,
            k_bshd,
            v_bshd,
            0,
            None,
            0.0,
            True,
            None,
        )
        return output.transpose(1, 2).contiguous()


def test_in_tree_custom_sdpa_vgf_lowers_to_tosa_custom():
    ensure_custom_sdpa_registered()
    q = torch.rand(1, 32, 1, 128)
    k = torch.rand(1, 8, 512, 128)
    v = torch.rand(1, 8, 512, 128)

    edge_model = to_edge(export(_DecodeCustomSDPA(), (q, k, v), strict=True))

    with TosaLoweringContext(TosaSpecification.create_from_string("TOSA-1.0+FP")):
        transformed = edge_model.transform([RewriteCustomSDPAToTosaCustomPass()])

    nodes = list(transformed.exported_program().graph.nodes)
    custom_node = next(
        node for node in nodes if node.target == exir_ops.backend.tosa.CUSTOM.default
    )

    assert custom_node.kwargs["operator_name"] == custom_sdpa_operator_name()
    assert custom_node.kwargs["domain_name"] == CUSTOM_SDPA_DOMAIN_NAME

    payload = decode_payload(custom_node.kwargs["implementation_attrs"])
    assert payload["layout"] == "BHSD"
    assert payload["query_shape"] == [1, 32, 1, 128]
    assert payload["key_shape"] == [1, 8, 512, 128]
    assert payload["value_shape"] == [1, 8, 512, 128]
    assert payload["is_causal"] is True
