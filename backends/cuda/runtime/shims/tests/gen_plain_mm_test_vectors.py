#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regenerate the hardcoded INT4/INT5/INT6 plain_mm dp4a test vectors.

This script deterministically recreates every ``uint8_t``/``int8_t``/``uint16_t``
array embedded in the plain_mm gtest files:
  --dtype int4 -> test_aoti_torch_cuda_int4_plain_mm.cpp
  --dtype int5 -> test_aoti_torch_cuda_int5_plain_mm.cpp
  --dtype int6 -> test_aoti_torch_cuda_int6_plain_mm.cpp
Each dtype has a fixed seed, so the emitted vectors are reproducible by
construction: the vectors in the .cpp are exactly this script's output.

All dtypes share the same scaffolding (a ``Case`` dataclass, C++ array emit,
the ``--check`` regression, and stdout emission). They differ only in the encode
math and the array field names:

INT4 (W4A8, asymmetric; pack path in coalesced_int4_tensor.py):
  1. torch.manual_seed(case.seed) on CPU.
  2. Draw a random bf16 weight ``[N, K]`` then activation ``[M, K]`` (weight
     first, then activation: fixed order is part of the seed contract).
  3. quantize_weight(..., bits=4, min_max, asymmetric) -> torchao Int4Tensor.
  4. CudaCoalescedInt4Tensor.from_int4_tensor(...) -> qdata [N, K/2] uint8,
     scale codes [N, K/gs] uint8, scale_step [N, K/256] fp16, zero_point codes
     [N, K/gs] uint8, zero_point_step [N, K/256] fp16.
  5. expected = F.linear(A, tensor.dequantize(bf16)).

INT5 (W5A8, asymmetric Q5_K; pack path in dp4a_planar_int5_tensor.py):
  1. torch.manual_seed(INT5_SEED) ONCE, then draw ALL cases in list order
     (unlike int4/int6, the int5 cases share one RNG stream, so build_int5
     replays the earlier cases to stay reproducible per-case).
  2. Per case, draw a scaled fp32 weight ``[N, K]`` (``randn * (0.5 +
     rand[N,1])``) then activation ``[M, K]`` (bf16). Weight-then-activation
     draw order is part of the seed contract.
  3. Quantize with the Q5_K affine min/max (asymmetric, u in [0, 31] centered to
     qdata in [-16, 15]) into an IntxUnpackedToInt8Tensor, then
     CudaDp4aPlanarInt5Tensor._from_intx_int8(...) -> ql [N, K/2] uint8, qh
     [N, K/8] uint8, scale codes [N, K/gs] uint8, scale_step [N, K/256] fp16,
     zero_point codes [N, K/gs] uint8, zero_point_step [N, K/256] fp16.
  4. expected = F.linear(A, tensor.dequantize(bf16)).

INT6 (W6A8, symmetric Q6_K, NO zero tensor; pack path in
dp4a_planar_int6_tensor.py):
  1. torch.manual_seed(case.seed) on CPU.
  2. Draw symmetric Q6_K values q ``[N, K]`` in [-32, 31], a small positive
     per-group scale ``[N, K/gs]`` (bf16), then an activation ``A`` ``[M, K]``
     (bf16). Draw order q -> scale -> A is part of the seed contract.
  3. pack_int6(q) -> planar ql [N, K/2] uint8, qh [N, K/4] uint8.
  4. _encode_int8_per_super(scale, gs) -> scale codes [N, K/gs] int8 +
     per-256-super-block step [N, K/256] fp16 (scale = code * step[:, g//gps]).
  5. expected = F.linear(A, tensor.dequantize(bf16)).

Both kernels quantize activations to int8, so the .cpp compares with a 0.5 atol.

Usage (from the executorch repo root, conda env with torch + torchao):
  python backends/cuda/runtime/shims/tests/gen_plain_mm_test_vectors.py \\
        --dtype {int4,int5,int6} [--case NAME] [--check]

Without ``--check`` it prints the C++ array blocks for each case to stdout; paste
them into the matching TEST_F body. With ``--check`` it re-derives the vectors
and asserts they match the constants currently in the .cpp (fast regression that
the .cpp was not hand-edited away from this generator).
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import torch


@dataclass(frozen=True)
class Case:
    name: str  # matches the TEST_F name in the .cpp
    M: int
    K: int
    N: int
    gs: int
    seed: int


# One entry per numerical TEST_F in each .cpp. The seeds are arbitrary but fixed:
# they define the checked-in vectors.
INT4_CASES: List[Case] = [
    Case("SingleSuperBlock", M=1, K=256, N=8, gs=32, seed=0),
    Case("MultiSuperBlock", M=1, K=512, N=4, gs=32, seed=1),
    Case("WideN", M=1, K=256, N=16, gs=32, seed=2),
    Case("PackedShuffleMultiSuper", M=1, K=1024, N=8, gs=32, seed=3),
]

# The INT5 cases share ONE RNG stream: torch.manual_seed(INT5_SEED) is called
# once and the cases are drawn in this list order (weight then activation per
# case). ``build_int5`` replays the earlier cases so each case stays reproducible
# on its own. All entries therefore carry the same seed.
INT5_SEED = 1234
INT5_CASES: List[Case] = [
    Case("SingleSuperBlock", M=1, K=256, N=8, gs=32, seed=INT5_SEED),
    Case("MultiSuperBlock", M=1, K=512, N=4, gs=32, seed=INT5_SEED),
    Case("WideN", M=1, K=256, N=16, gs=32, seed=INT5_SEED),
    Case("PackedShuffleMultiSuper", M=1, K=1024, N=8, gs=32, seed=INT5_SEED),
]

INT6_CASES: List[Case] = [
    Case("Q6KSingleSuperBlock", M=2, K=256, N=4, gs=16, seed=0),
    Case("Q6KMultiSuperBlock", M=1, K=512, N=6, gs=16, seed=1),
    Case("Q6KWideN", M=1, K=256, N=16, gs=16, seed=2),
]


# ---------------------------------------------------------------------------
# Shared bit/format helpers.
# ---------------------------------------------------------------------------
def _fp16_bits(t: torch.Tensor) -> List[int]:
    """fp16 tensor -> list of uint16 bit patterns (little-endian half bits)."""
    return (
        t.to(torch.float16)
        .contiguous()
        .view(torch.int16)
        .to(torch.int32)
        .bitwise_and(0xFFFF)
        .flatten()
        .tolist()
    )


def _bf16_bits(t: torch.Tensor) -> List[int]:
    """bf16 tensor -> list of uint16 bit patterns."""
    return (
        t.to(torch.bfloat16)
        .contiguous()
        .view(torch.int16)
        .to(torch.int32)
        .bitwise_and(0xFFFF)
        .flatten()
        .tolist()
    )


def _u8(t: torch.Tensor) -> List[int]:
    return t.to(torch.uint8).contiguous().flatten().tolist()


def _i8(t: torch.Tensor) -> List[int]:
    return t.to(torch.int8).contiguous().flatten().tolist()


# ---------------------------------------------------------------------------
# Dtype-specific encode paths. Imports are lazy so generating one dtype does not
# require the other dtype's dependencies.
# ---------------------------------------------------------------------------
def build_int4(case: Case) -> Dict[str, tuple]:
    """Return {array_name: (ctype, [ints])} for one INT4 case (CPU only)."""
    from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor
    from executorch.examples.models.gemma4_31b.quant.quantize import quantize_weight
    from executorch.examples.models.gemma4_31b.quant.recipe import QuantConfig

    torch.manual_seed(case.seed)
    # Weight first, then activation: fixed order is part of the seed contract.
    w = torch.randn(case.N, case.K, dtype=torch.bfloat16)
    A = torch.randn(case.M, case.K, dtype=torch.bfloat16)

    config = QuantConfig(bits=4, group_size=case.gs, symmetric=False, method="min_max")
    int4 = quantize_weight(w, config)
    c = CudaCoalescedInt4Tensor.from_int4_tensor(int4)

    # bf16 dequant @ F.linear reference (kernel adds activation-quant noise).
    w_deq = c.dequantize(torch.bfloat16)
    expected = torch.nn.functional.linear(A, w_deq)

    return {
        "qdata_host": ("uint8_t", _u8(c.qdata)),
        "scale_codes": ("uint8_t", _u8(c.scale)),
        "scale_step": ("uint16_t", _fp16_bits(c.scale_step)),
        "zero_codes": ("uint8_t", _u8(c.zero_point)),
        "zero_point_step": ("uint16_t", _fp16_bits(c.zero_point_step)),
        "A_host": ("uint16_t", _bf16_bits(A)),
        "expected": ("uint16_t", _bf16_bits(expected)),
    }


def _draw_int5_case(case: Case):
    """Consume the RNG for one INT5 case; return (IntxUnpackedToInt8Tensor, A).

    Weight then activation draw order is part of the seed contract. The weight is
    ``randn * (0.5 + rand[N,1])`` (a per-row scaled normal) quantized with the
    Q5_K affine min/max: u in [0, 31] centered to qdata in [-16, 15], with an
    asymmetric bf16 zero point folded from the group min (matches the real
    ``ExportableGGUFTensor.to_intx_unpacked_to_int8_tensor`` Q5_K branch).
    """
    from torchao.quantization import IntxUnpackedToInt8Tensor

    ng = case.K // case.gs
    w = torch.randn(case.N, case.K, dtype=torch.float32) * (0.5 + torch.rand(case.N, 1))
    A = torch.randn(case.M, case.K, dtype=torch.bfloat16)

    wg = w.reshape(case.N, ng, case.gs)
    wmin = wg.amin(dim=2)
    wmax = wg.amax(dim=2)
    eff_scale = ((wmax - wmin) / 31.0).clamp_min(1e-6)
    u = torch.round((wg - wmin.unsqueeze(-1)) / eff_scale.unsqueeze(-1)).clamp_(0, 31)
    u = u.to(torch.int16).reshape(case.N, case.K)
    zero = (-wmin / eff_scale).clamp_min(0.0)  # (N, ng) >= 0

    src = IntxUnpackedToInt8Tensor(
        qdata=(u - 16).to(torch.int8),  # center [0, 31] -> [-16, 15]
        scale=eff_scale.to(torch.bfloat16),
        zero_point=(zero - 16.0).to(torch.bfloat16),  # centered like gguf.py
        target_dtype=torch.int5,
        block_size=(1, case.gs),
        dtype=torch.bfloat16,
        activation_quantization=None,
    )
    return src, A


def build_int5(case: Case) -> Dict[str, tuple]:
    """Return {array_name: (ctype, [ints])} for one INT5 case (CPU only).

    The INT5 cases share one RNG stream (seed once, draw in ``INT5_CASES``
    order), so this seeds ``INT5_SEED`` and replays every earlier case's draw
    before building ``case``. That keeps each case independently reproducible
    through the per-case ``build`` interface while matching the checked-in .cpp.
    """
    from executorch.backends.cuda.dp4a_planar_int5_tensor import (
        CudaDp4aPlanarInt5Tensor,
    )

    idx = next(i for i, c in enumerate(INT5_CASES) if c.name == case.name)
    torch.manual_seed(case.seed)
    src = A = None
    for c in INT5_CASES[: idx + 1]:
        src, A = _draw_int5_case(c)

    tensor = CudaDp4aPlanarInt5Tensor._from_intx_int8(src)

    # bf16 dequant @ F.linear reference (kernel adds activation-quant noise).
    w_deq = tensor.dequantize(torch.bfloat16)
    expected = torch.nn.functional.linear(A, w_deq)

    return {
        "ql_host": ("uint8_t", _u8(tensor.ql)),
        "qh_host": ("uint8_t", _u8(tensor.qh)),
        "scale_codes": ("uint8_t", _u8(tensor.scale)),
        "scale_step": ("uint16_t", _fp16_bits(tensor.scale_step)),
        "zero_codes": ("uint8_t", _u8(tensor.zero_point)),
        "zero_point_step": ("uint16_t", _fp16_bits(tensor.zero_point_step)),
        "A_host": ("uint16_t", _bf16_bits(A)),
        "expected": ("uint16_t", _bf16_bits(expected)),
    }


def build_int6(case: Case) -> Dict[str, tuple]:
    """Return {array_name: (ctype, [ints])} for one INT6 case (CPU only)."""
    from executorch.backends.cuda.dp4a_planar_int6_tensor import (
        _encode_int8_per_super,
        CudaDp4aPlanarInt6Tensor,
        pack_int6,
    )

    torch.manual_seed(case.seed)
    # Symmetric q, then scale, then activation: fixed order is part of the seed
    # contract. Matches test_int6_dispatch._make_int6_tensor's convention.
    q = torch.randint(-32, 32, (case.N, case.K), dtype=torch.int8)
    scale = (torch.rand(case.N, case.K // case.gs) * 0.1 + 0.01).to(torch.bfloat16)
    A = torch.randn(case.M, case.K, dtype=torch.bfloat16)

    ql, qh = pack_int6(q)
    scale_codes, steps = _encode_int8_per_super(scale.float(), case.gs)
    tensor = CudaDp4aPlanarInt6Tensor(
        ql, qh, scale_codes, steps, [1, case.gs], torch.Size([case.N, case.K])
    )

    # bf16 dequant @ F.linear reference (kernel adds activation-quant noise).
    w_deq = tensor.dequantize(torch.bfloat16)
    expected = torch.nn.functional.linear(A, w_deq)

    return {
        "ql_host": ("uint8_t", _u8(ql)),
        "qh_host": ("uint8_t", _u8(qh)),
        "scale_codes": ("int8_t", _i8(scale_codes)),
        "scale_step": ("uint16_t", _fp16_bits(steps)),
        "A_host": ("uint16_t", _bf16_bits(A)),
        "expected": ("uint16_t", _bf16_bits(expected)),
    }


@dataclass(frozen=True)
class DtypeSpec:
    name: str
    cases: List[Case]
    order: List[str]  # array emit order
    build: Callable[[Case], Dict[str, tuple]]
    test_class: str  # TEST_F fixture class name in the .cpp
    cpp_name: str  # default .cpp filename (same dir as this script)


SPECS: Dict[str, DtypeSpec] = {
    "int4": DtypeSpec(
        name="int4",
        cases=INT4_CASES,
        order=[
            "qdata_host",
            "scale_codes",
            "scale_step",
            "zero_codes",
            "zero_point_step",
            "A_host",
            "expected",
        ],
        build=build_int4,
        test_class="AOTITorchInt4PlainMMTest",
        cpp_name="test_aoti_torch_cuda_int4_plain_mm.cpp",
    ),
    "int5": DtypeSpec(
        name="int5",
        cases=INT5_CASES,
        order=[
            "ql_host",
            "qh_host",
            "scale_codes",
            "scale_step",
            "zero_codes",
            "zero_point_step",
            "A_host",
            "expected",
        ],
        build=build_int5,
        test_class="AOTITorchInt5PlainMMTest",
        cpp_name="test_aoti_torch_cuda_int5_plain_mm.cpp",
    ),
    "int6": DtypeSpec(
        name="int6",
        cases=INT6_CASES,
        order=["ql_host", "qh_host", "scale_codes", "scale_step", "A_host", "expected"],
        build=build_int6,
        test_class="AOTITorchInt6PlainMMTest",
        cpp_name="test_aoti_torch_cuda_int6_plain_mm.cpp",
    ),
}


# ---------------------------------------------------------------------------
# Shared emit / check scaffolding.
# ---------------------------------------------------------------------------
def _fmt_array(name: str, ctype: str, values: List[int]) -> str:
    if ctype == "uint8_t":
        per_line, cell = 12, lambda v: f"0x{v & 0xFF:02X}"
    elif ctype == "int8_t":
        # Signed decimal, right-aligned like the .cpp (Q6_K scale codes).
        per_line, cell = 12, lambda v: f"{v:4d}"
    elif ctype == "uint16_t":
        per_line, cell = 8, lambda v: f"0x{v & 0xFFFF:04X}"
    else:
        raise ValueError(f"unsupported ctype {ctype}")
    lines = [f"  {ctype} {name}[] = {{"]
    for i in range(0, len(values), per_line):
        cells = ", ".join(cell(v) for v in values[i : i + per_line])
        lines.append(f"      {cells},")
    lines.append("  };")
    return "\n".join(lines)


def emit(spec: DtypeSpec, case: Case) -> str:
    vecs = spec.build(case)
    blocks = [
        f"// ==== {case.name} (M={case.M}, K={case.K}, N={case.N}, "
        f"gs={case.gs}, seed={case.seed}) ===="
    ]
    for name in spec.order:
        ctype, values = vecs[name]
        blocks.append(_fmt_array(name, ctype, values))
    return "\n".join(blocks)


def _parse_cpp_array(text: str, test_class: str, case_name: str, arr: str) -> List[int]:
    """Extract a single array's ints from the given TEST_F body in the .cpp.

    Handles both hex cells (e.g. ql/qh/scale_step/A/expected) and signed-decimal
    cells (int6 scale_codes int8).
    """
    m = re.search(
        rf"TEST_F\({re.escape(test_class)},\s*{re.escape(case_name)}\)",
        text,
    )
    if not m:
        raise AssertionError(f"TEST_F {case_name} not found in .cpp")
    body = text[m.end() :]
    pattern = r"\b" + re.escape(arr) + r"\[\]\s*=\s*\{(.*?)\};"
    am = re.search(pattern, body, re.DOTALL)
    if not am:
        raise AssertionError(f"array {arr} not found in TEST_F {case_name}")
    return [int(x, 0) for x in re.findall(r"-?0[xX][0-9a-fA-F]+|-?\d+", am.group(1))]


def check(spec: DtypeSpec, cpp_path: Path) -> None:
    text = cpp_path.read_text()
    for case in spec.cases:
        vecs = spec.build(case)
        for arr, (_ctype, values) in vecs.items():
            got = _parse_cpp_array(text, spec.test_class, case.name, arr)
            assert got == values, (
                f"{case.name}:{arr} mismatch vs .cpp "
                f"(script has {len(values)} vals, .cpp has {len(got)}). "
                "Regenerate the .cpp from this script."
            )
    print(f"OK: all checked-in {spec.name} vectors match the generator output.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dtype",
        required=True,
        choices=sorted(SPECS),
        help="which plain_mm vectors to emit/check",
    )
    ap.add_argument("--case", help="only emit/check this TEST_F case")
    ap.add_argument(
        "--check",
        action="store_true",
        help="verify the .cpp constants match the generator instead of printing",
    )
    ap.add_argument(
        "--cpp",
        default=None,
        help="path to the .cpp test (for --check); defaults per --dtype",
    )
    args = ap.parse_args()

    spec = SPECS[args.dtype]
    cpp_path = Path(args.cpp) if args.cpp else Path(__file__).with_name(spec.cpp_name)

    cases = spec.cases
    if args.case:
        cases = [c for c in spec.cases if c.name == args.case]
        if not cases:
            raise SystemExit(f"unknown {args.dtype} case {args.case!r}")

    if args.check:
        # --check always validates the full set (mismatched subset is confusing).
        check(spec, cpp_path)
        return

    for case in cases:
        print(emit(spec, case))
        print()


if __name__ == "__main__":
    main()
