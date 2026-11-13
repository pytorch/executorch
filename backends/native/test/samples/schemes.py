# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Quantization schemes. Every op is recorded under every scheme, keeping the
cases the scheme fits."""

import contextlib
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import torch
from executorch.extension.llm.export.gguf import ExportableGGUFTensor, iter_gguf, QK_K
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFWriter
from torchao.core.config import AOBaseConfig
from torchao.quantization import (
    dequantize_affine,
    Int8DynamicActivationIntxWeightConfig,
    IntxUnpackedToInt8Tensor,
    IntxWeightOnlyConfig,
    quantize_,
)
from torchao.quantization.granularity import PerGroup

# Private, but it is the input quantize-dequantize that
# Int8DynamicActivationIntxWeightConfig linears run.
from torchao.quantization.quantize_.workflows.intx.intx_unpacked_to_int8_tensor import (
    _apply_int8_act_asym_per_token_quant_dequant,
)
from torchao.utils import TorchAOBaseTensor


class Scheme:
    """No quantization."""

    @property
    def weight_only(self) -> bool:
        """Whether the scheme quantizes only weights; `record` then adds
        `golden-int8_act`."""
        return False

    def fits(self, model: torch.nn.Module) -> bool:
        return True

    def quantize(self, model: torch.nn.Module) -> None:
        pass

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        """A state tensor of a quantized model as a plain tensor, with quantized
        weights decoded in float32 rather than the method's dtype, which can
        round them."""
        return value


# Modules whose weight a weight-only scheme quantizes, each grouped along its
# rows: an nn.Linear's in_features, an nn.Embedding's embedding_dim.
_WO_QSCHEME_MODULES = (torch.nn.Linear, torch.nn.Embedding)
# Modules whose inputs the int8 dynamic activation schemes quantize.
_DQA_QSCHEME_MODULES = (torch.nn.Linear,)


def _targets(model: torch.nn.Module, types: tuple[type, ...]) -> list[torch.nn.Module]:
    return [m for m in model.modules() if isinstance(m, types)]


def _fits(model: torch.nn.Module, types: tuple[type, ...], group: int | None) -> bool:
    """Whether `model` has a module of `types` and every one's weight rows split
    into groups of `group`, or form one group when it is None."""
    targets = _targets(model, types)
    for m in targets:
        width = m.weight.shape[1]
        size = width if group is None else group
        # torchao reduces a one-element-group weight to a single scale.
        if size < 2 or width % size:
            return False
    return bool(targets)


@dataclass(frozen=True)
class Torchao(Scheme):
    """A torchao config applied to every nn.Linear and, when it quantizes only
    weights, every nn.Embedding."""

    config: AOBaseConfig

    @property
    def weight_only(self) -> bool:
        return isinstance(self.config, IntxWeightOnlyConfig)

    @property
    def _types(self) -> tuple[type, ...]:
        return _WO_QSCHEME_MODULES if self.weight_only else _DQA_QSCHEME_MODULES

    def fits(self, model: torch.nn.Module) -> bool:
        granularity = (
            self.config.granularity
            if isinstance(self.config, IntxWeightOnlyConfig)
            else self.config.weight_granularity
        )
        return _fits(
            model,
            self._types,
            granularity.group_size if isinstance(granularity, PerGroup) else None,
        )

    def quantize(self, model: torch.nn.Module) -> None:
        quantize_(model, self.config, filter_fn=lambda m, _: isinstance(m, self._types))

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        if isinstance(value, IntxUnpackedToInt8Tensor):
            return dequantize_affine(
                value.qdata, value.block_size, value.scale, value.zero_point, torch.int8
            )
        if isinstance(value, TorchAOBaseTensor):
            raise TypeError(f"no float32 decode for {type(value).__name__}")
        return value


# Families of fp16 super-block scales by byte offset in each block: powers of
# two, where a dmin of 8 d (q4_k) or 16 d (q5_k) centers random weights near
# zero, then negative, rounded and subnormal ones. A model's i-th quantized
# module takes the i-th family, so one module per family isolates each in its
# own output.
# TODO: Add a zero-d family once OutputCheck's bias standard error stops
# assuming independent outputs.
GGUF_SUPER_SCALES = {
    "q4_k": [
        {0: 2**-12, 2: 2**-9},
        {0: -(2**-12), 2: -(2**-9)},
        {0: 3.1e-4, 2: 2.3e-3},
        {0: 3 * 2**-21, 2: 3 * 2**-18},
    ],
    "q5_k": [
        {0: 2**-12, 2: 2**-8},
        {0: -(2**-12), 2: -(2**-8)},
        {0: 3.1e-4, 2: 4.8e-3},
        {0: 3 * 2**-21, 2: 3 * 2**-17},
    ],
    "q6_k": [
        {208: 2**-12},
        {208: -(2**-12)},
        {208: 3.1e-4},
        {208: 3 * 2**-21},
    ],
}


def _random_blocks(
    ggml_type: str,
    rows: int,
    cols: int,
    scales: dict[int, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Valid `ggml_type` blocks for a [rows, cols] weight from random bytes and
    the given super-block scales."""
    _, block_bytes = GGML_QUANT_SIZES[GGMLQuantizationType[ggml_type.upper()]]
    raw = rng.integers(0, 256, (rows, cols // QK_K, block_bytes), dtype=np.uint8)
    for offset, scale in scales.items():
        raw[..., offset : offset + 2] = np.array([scale], np.float16).view(np.uint8)
    return raw.reshape(rows, -1)


def _load_gguf(model: torch.nn.Module, ggml_type: str) -> None:
    """Writes random `ggml_type` weights for every nn.Linear and nn.Embedding to
    a .gguf file and loads them back as a GGUF checkpoint would be."""
    targets = _targets(model, _WO_QSCHEME_MODULES)
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/weights.gguf"
        writer = GGUFWriter(path, arch="samples")
        families = GGUF_SUPER_SCALES[ggml_type]
        if len(targets) > len(families):
            raise ValueError(
                f"{len(targets)} weights, {len(families)} {ggml_type} scale families"
            )
        for i, (m, scales) in enumerate(zip(targets, families)):
            writer.add_tensor(
                f"weight.{i}",
                _random_blocks(ggml_type, *m.weight.shape, scales, rng),
                raw_dtype=GGMLQuantizationType[ggml_type.upper()],
            )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        for m, (_, weight) in zip(targets, iter_gguf(path), strict=True):
            m.weight = torch.nn.Parameter(
                weight.to(m.weight.dtype), requires_grad=False
            )


@dataclass(frozen=True)
class Gguf(Scheme):
    """Weight-only GGUF k-quant blocks (`q4_k`, `q5_k` or `q6_k`) for every
    nn.Linear and nn.Embedding, loaded from a .gguf file."""

    ggml_type: str

    @property
    def weight_only(self) -> bool:
        return True

    def fits(self, model: torch.nn.Module) -> bool:
        families = len(GGUF_SUPER_SCALES[self.ggml_type])
        return (
            _fits(model, _WO_QSCHEME_MODULES, QK_K)
            and len(_targets(model, _WO_QSCHEME_MODULES)) <= families
        )

    def quantize(self, model: torch.nn.Module) -> None:
        _load_gguf(model, self.ggml_type)

    def decode(self, value: torch.Tensor) -> torch.Tensor:
        if isinstance(value, ExportableGGUFTensor):
            return value.dequantize(torch.float32)
        return value


@contextlib.contextmanager
def int8_activations(model: torch.nn.Module) -> Iterator[None]:
    """Quantizes the inputs of every `_DQA_QSCHEME_MODULES` module as torchao's
    int8 dynamic activation schemes do: int8, asymmetric, per token."""
    hooks = [
        m.register_forward_pre_hook(
            lambda _, args: (_apply_int8_act_asym_per_token_quant_dequant(args[0]),)
        )
        for m in model.modules()
        if isinstance(m, _DQA_QSCHEME_MODULES)
    ]
    try:
        yield
    finally:
        for hook in hooks:
            hook.remove()


def schemes() -> dict[str, Scheme]:
    return {
        "float": Scheme(),
        "int4wo_g32": Torchao(
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(32))
        ),
        "int4wo_g128": Torchao(
            IntxWeightOnlyConfig(weight_dtype=torch.int4, granularity=PerGroup(128))
        ),
        "8da4w_g32": Torchao(
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4, weight_granularity=PerGroup(32)
            )
        ),
        "8da4w_g128": Torchao(
            Int8DynamicActivationIntxWeightConfig(
                weight_dtype=torch.int4, weight_granularity=PerGroup(128)
            )
        ),
        "gguf_q4_k": Gguf("q4_k"),
        "gguf_q5_k": Gguf("q5_k"),
        "gguf_q6_k": Gguf("q6_k"),
    }
