# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json

import torch
import torch.nn.functional as F
from executorch.backends.arm.tosa.dialect.ops.custom import register_fake_tosa
from executorch.backends.arm.vgf.shaders.sdpa import build_custom_sdpa_shader_payload

CUSTOM_SDPA_DOMAIN_NAME = "com.arm.VulkanCustomShader"
_CUSTOM_SDPA_OPERATOR_NAME = "llama.custom_sdpa"
_PYTHON_CUSTOM_SDPA_LIB: torch.library.Library | None = None


def custom_sdpa_operator_name() -> str:
    """Return the stable TOSA custom-operator identifier for SDPA."""
    return _CUSTOM_SDPA_OPERATOR_NAME


def _has_custom_sdpa_registration() -> bool:
    try:
        return torch.ops.llama.custom_sdpa.default is not None
    except (AttributeError, RuntimeError):
        return False


def _custom_sdpa_output_shape(
    query: torch.Tensor, value: torch.Tensor
) -> tuple[int, ...]:
    return tuple(query.shape[:-1]) + (value.shape[-1],)


def _register_python_custom_sdpa() -> None:
    global _PYTHON_CUSTOM_SDPA_LIB
    if _has_custom_sdpa_registration() or _PYTHON_CUSTOM_SDPA_LIB is not None:
        return

    _PYTHON_CUSTOM_SDPA_LIB = torch.library.Library("llama", "FRAGMENT")
    _PYTHON_CUSTOM_SDPA_LIB.define(
        "custom_sdpa("
        "Tensor query, Tensor key, Tensor value, SymInt start_pos, "
        "Tensor? attn_mask=None, float drpout_p=0.0, bool is_causal=False, "
        "float? scale=None) -> Tensor"
    )

    @torch.library.impl(
        _PYTHON_CUSTOM_SDPA_LIB,
        "custom_sdpa",
        "CompositeExplicitAutograd",
    )
    def _custom_sdpa_python_impl(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos: int,
        attn_mask: torch.Tensor | None = None,
        drpout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        runtime_is_causal = is_causal
        if is_causal:
            key, value = _slice_active_prefix(
                key,
                value,
                layout="BSHD",
                start_pos=start_pos,
            )
            runtime_is_causal = False
        return _run_decode_sdpa_bshd(
            query,
            key,
            value,
            attn_mask,
            drpout_p,
            runtime_is_causal,
            scale,
        )

    @torch.library.impl(_PYTHON_CUSTOM_SDPA_LIB, "custom_sdpa", "Meta")
    def _custom_sdpa_meta_impl(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        start_pos: int,
        attn_mask: torch.Tensor | None = None,
        drpout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
    ) -> torch.Tensor:
        _ = (key, start_pos, attn_mask, drpout_p, is_causal, scale)
        return torch.empty(
            _custom_sdpa_output_shape(query, value),
            dtype=query.dtype,
            device=query.device,
        )


def ensure_custom_sdpa_registered() -> None:
    """Load a ``llama.custom_sdpa`` registration path when needed."""
    if _has_custom_sdpa_registration():
        return

    try:
        from executorch.extension.llm.custom_ops import custom_ops  # noqa: F401
    except (ImportError, OSError, RuntimeError):
        _register_python_custom_sdpa()


def build_custom_sdpa_payload(
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    query_dtype: torch.dtype,
    key_dtype: torch.dtype,
    value_dtype: torch.dtype,
    start_pos: int,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
    *,
    layout: str = "BSHD",
) -> dict[str, object]:
    """Build the serialized payload for the Arm SDPA ``tosa.CUSTOM`` path."""
    if query_dtype != torch.float32:
        raise ValueError(
            "decode SDPA shader payload only supports float32 query tensors"
        )
    if key_dtype != torch.float32:
        raise ValueError("decode SDPA shader payload only supports float32 key tensors")
    if value_dtype != torch.float32:
        raise ValueError(
            "decode SDPA shader payload only supports float32 value tensors"
        )
    return build_custom_sdpa_shader_payload(
        query_shape,
        key_shape,
        value_shape,
        start_pos,
        dropout_p,
        is_causal,
        scale,
        layout=layout,
    )


def encode_payload(payload: dict[str, object]) -> list[int]:
    """Encode the SDPA custom payload as ``implementation_attrs`` bytes."""
    return list(json.dumps(payload, sort_keys=True).encode("utf-8"))


def decode_payload(implementation_attrs: list[int]) -> dict[str, object]:
    """Decode ``implementation_attrs`` bytes for the SDPA custom payload."""
    return json.loads(bytes(implementation_attrs).decode("utf-8"))


def _run_decode_sdpa_bshd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=True,
    ).transpose(1, 2)


def _run_decode_sdpa_bhsd(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    dropout_p: float,
    is_causal: bool,
    scale: float | None,
) -> torch.Tensor:
    return F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attn_mask,
        dropout_p=dropout_p,
        is_causal=is_causal,
        scale=scale,
        enable_gqa=True,
    )


def _slice_active_prefix(
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    layout: str,
    start_pos: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    active_sequence_length = min(
        key.shape[1] if layout == "BSHD" else key.shape[2],
        start_pos + 1,
    )
    if layout == "BSHD":
        return (
            key[:, :active_sequence_length, :, :],
            value[:, :active_sequence_length, :, :],
        )
    return (
        key[:, :, :active_sequence_length, :],
        value[:, :, :active_sequence_length, :],
    )


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
    start_pos = payload.get("start_pos", 0)
    scale = payload["scale"]
    dropout_p = payload["dropout_p"]
    is_causal = payload["is_causal"]
    attn_mask = inputs[3] if len(inputs) == 4 else None
    assert isinstance(dropout_p, (float, int))
    assert isinstance(is_causal, bool)
    assert isinstance(layout, str)
    assert scale is None or isinstance(scale, (float, int))
    assert isinstance(start_pos, int)

    assert operator_name == custom_sdpa_operator_name()
    assert domain_name == CUSTOM_SDPA_DOMAIN_NAME
    key = inputs[1]
    value = inputs[2]
    runtime_is_causal = is_causal
    if is_causal:
        key, value = _slice_active_prefix(
            key,
            value,
            layout=layout,
            start_pos=start_pos,
        )
        runtime_is_causal = False
    if layout == "BSHD":
        output = _run_decode_sdpa_bshd(
            inputs[0],
            key,
            value,
            attn_mask,
            float(dropout_p),
            runtime_is_causal,
            None if scale is None else float(scale),
        )
    elif layout == "BHSD":
        output = _run_decode_sdpa_bhsd(
            inputs[0],
            key,
            value,
            attn_mask,
            float(dropout_p),
            runtime_is_causal,
            None if scale is None else float(scale),
        )
    else:
        raise RuntimeError(f"Unsupported SDPA payload layout: {layout}")
    return [output]
