# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import torch
from executorch.backends.arm.tosa.dialect.ops.custom import register_fake_tosa

CUSTOM_SDPA_DOMAIN_NAME = "com.arm.VulkanCustomOp"
_CUSTOM_SDPA_OPERATOR_NAME = "llama.custom_sdpa"


def custom_sdpa_operator_name() -> str:
    """Return the stable TOSA custom-operator identifier for SDPA."""
    return _CUSTOM_SDPA_OPERATOR_NAME


def ensure_custom_sdpa_registered() -> None:
    """Load a ``llama.custom_sdpa`` registration path when needed."""
    from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401


def build_custom_sdpa_payload(
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    start_pos: int,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    *,
    layout: str = "BSHD",
) -> dict[str, object]:
    """Build the serialized payload for the Arm SDPA ``tosa.CUSTOM`` path."""
    return {
        "api_version": 1,
        "layout": layout,
        "query_shape": list(query_shape),
        "key_shape": list(key_shape),
        "value_shape": list(value_shape),
        "start_pos": int(start_pos),
        "dropout_p": float(dropout_p),
        "is_causal": bool(is_causal),
        "scale": None if scale is None else float(scale),
    }


def encode_payload(payload: dict[str, object]) -> list[int]:
    """Encode the SDPA custom payload as ``implementation_attrs`` bytes."""
    return list(json.dumps(payload, sort_keys=True).encode("utf-8"))


def decode_payload(implementation_attrs: list[int]) -> dict[str, object]:
    """Decode ``implementation_attrs`` bytes for the SDPA custom payload."""
    return json.loads(bytes(implementation_attrs).decode("utf-8"))


@register_fake_tosa(custom_sdpa_operator_name())
def custom_sdpa_tosa_fake_impl(
    inputs: list[torch.Tensor],
    operator_name: str,
    domain_name: str,
    implementation_attrs: list[int],
) -> list[torch.Tensor]:
    """Run the compiler-side fake implementation for SDPA ``tosa.CUSTOM``."""
    ensure_custom_sdpa_registered()

    if len(inputs) not in (3, 4):
        raise RuntimeError(
            f"{custom_sdpa_operator_name()} expects 3 or 4 tensor inputs, got "
            f"{len(inputs)}"
        )

    payload = decode_payload(implementation_attrs)
    layout = payload.get("layout", "BSHD")
    start_pos = payload["start_pos"]
    scale = payload["scale"]
    dropout_p = payload["dropout_p"]
    is_causal = payload["is_causal"]
    attn_mask = inputs[3] if len(inputs) == 4 else None
    assert isinstance(layout, str)
    assert isinstance(start_pos, int)
    assert isinstance(dropout_p, (float, int))
    assert isinstance(is_causal, bool)
    assert scale is None or isinstance(scale, (float, int))
    query = inputs[0]
    key = inputs[1]
    value = inputs[2]
    if layout == "BHSD":
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

    assert operator_name == custom_sdpa_operator_name()
    assert domain_name == CUSTOM_SDPA_DOMAIN_NAME
    output = torch.ops.llama.custom_sdpa.default(
        query,
        key,
        value,
        start_pos,
        attn_mask,
        float(dropout_p),
        is_causal,
        None if scale is None else float(scale),
    )
    if layout == "BHSD":
        output = output.permute(0, 2, 1, 3).contiguous()
    return [output]
