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
from executorch.extension.llm.export.gguf import ExportableGGUFTensor, iter_gguf
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
        """Whether the scheme quantizes only weights."""
        return False

    def adds_int8_act(self, model: torch.nn.Module) -> bool:
        """Whether `record` adds `golden-int8_act` for `model`: the scheme
        quantizes only weights and `model` has inputs int8_activations
        quantizes."""
        return self.weight_only and bool(_targets(model, _DQA_QSCHEME_MODULES))

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


# Families of fp16 block scales by byte offset in each block: powers of two,
# where a type's dmin, if it has one, centers random weights near zero, then
# negative, rounded and subnormal ones, and larger powers of two for a fifth
# weight, as a Gated DeltaNet layer has. A model's i-th quantized module takes
# the i-th family, so one module per family isolates each in its own output.
# TODO: Add a zero-d family once OutputCheck's bias standard error stops
# assuming independent outputs.
GGUF_SUPER_SCALES = {
    "q4_0": [
        {0: 2**-12},
        {0: -(2**-12)},
        {0: 3.1e-4},
        {0: 3 * 2**-21},
        {0: 2**-10},
    ],
    "q4_1": [
        {0: 2**-12, 2: -15 * 2**-13},
        {0: -(2**-12), 2: 15 * 2**-13},
        {0: 3.1e-4, 2: -2.3e-3},
        {0: 3 * 2**-21, 2: -45 * 2**-22},
        {0: 2**-10, 2: -15 * 2**-11},
    ],
    "q5_0": [
        {0: 2**-12},
        {0: -(2**-12)},
        {0: 3.1e-4},
        {0: 3 * 2**-21},
        {0: 2**-10},
    ],
    "q5_1": [
        {0: 2**-12, 2: -31 * 2**-13},
        {0: -(2**-12), 2: 31 * 2**-13},
        {0: 3.1e-4, 2: -4.8e-3},
        {0: 3 * 2**-21, 2: -93 * 2**-22},
        {0: 2**-10, 2: -31 * 2**-11},
    ],
    "q8_0": [
        {0: 2**-12},
        {0: -(2**-12)},
        {0: 3.1e-4},
        {0: 3 * 2**-21},
        {0: 2**-10},
    ],
    "q2_k": [
        {80: 2**-12, 82: 3 * 2**-13},
        {80: -(2**-12), 82: -3 * 2**-13},
        {80: 3.1e-4, 82: 4.7e-4},
        {80: 3 * 2**-21, 82: 9 * 2**-22},
        {80: 2**-10, 82: 3 * 2**-11},
    ],
    "q3_k": [
        {108: 2**-12},
        {108: -(2**-12)},
        {108: 3.1e-4},
        {108: 3 * 2**-21},
        {108: 2**-10},
    ],
    "q4_k": [
        {0: 2**-12, 2: 2**-9},
        {0: -(2**-12), 2: -(2**-9)},
        {0: 3.1e-4, 2: 2.3e-3},
        {0: 3 * 2**-21, 2: 3 * 2**-18},
        {0: 2**-10, 2: 2**-7},
    ],
    "q5_k": [
        {0: 2**-12, 2: 2**-8},
        {0: -(2**-12), 2: -(2**-8)},
        {0: 3.1e-4, 2: 4.8e-3},
        {0: 3 * 2**-21, 2: 3 * 2**-17},
        {0: 2**-10, 2: 2**-6},
    ],
    "q6_k": [
        {208: 2**-12},
        {208: -(2**-12)},
        {208: 3.1e-4},
        {208: 3 * 2**-21},
        {208: 2**-10},
    ],
}


def _block(ggml_type: str) -> tuple[int, int]:
    """Weights and bytes per block of a GGUF quant type."""
    return GGML_QUANT_SIZES[GGMLQuantizationType[ggml_type.upper()]]


def _random_blocks(
    ggml_type: str,
    rows: int,
    cols: int,
    scales: dict[int, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Valid `ggml_type` blocks for a [rows, cols] weight from random bytes and
    the given block scales."""
    block, block_bytes = _block(ggml_type)
    raw = rng.integers(0, 256, (rows, cols // block, block_bytes), dtype=np.uint8)
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
    """Weight-only GGUF blocks of `ggml_type` (`q4_k`, `q8_0`, ...) for every
    nn.Linear and nn.Embedding, loaded from a .gguf file."""

    ggml_type: str

    @property
    def weight_only(self) -> bool:
        return True

    def fits(self, model: torch.nn.Module) -> bool:
        families = len(GGUF_SUPER_SCALES[self.ggml_type])
        return (
            _fits(model, _WO_QSCHEME_MODULES, _block(self.ggml_type)[0])
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
        "gguf_q4_0": Gguf("q4_0"),
        "gguf_q4_1": Gguf("q4_1"),
        "gguf_q5_0": Gguf("q5_0"),
        "gguf_q5_1": Gguf("q5_1"),
        "gguf_q8_0": Gguf("q8_0"),
        "gguf_q2_k": Gguf("q2_k"),
        "gguf_q3_k": Gguf("q3_k"),
        "gguf_q4_k": Gguf("q4_k"),
        "gguf_q5_k": Gguf("q5_k"),
        "gguf_q6_k": Gguf("q6_k"),
    }
