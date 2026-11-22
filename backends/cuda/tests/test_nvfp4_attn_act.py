# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-weight activation format of NVFP4 linears (``CudaNVFP4Tensor.act_dtype``).

mxfp8: MXFP8 x MXFP8 prefill on GPUs with FP4 conversion; elsewhere BF16
activations by default or, with ET_CUDA_NVFP4_MXFP8_EMULATE=1, an MXFP8-grid
emulation. W4A16 or INT8-activation W4A8 decode.
bf16: weight-only W4A16. Neither may ever reach a W4A4 kernel.
"""

import os
import unittest
from unittest import mock

import executorch.backends.cuda.quantize_op_dispatch  # noqa: F401

import torch
import torch.nn.functional as F
import triton
from executorch.backends.cuda.nvfp4_tensor import CudaNVFP4Tensor
from executorch.backends.cuda.quantize_op_dispatch import nvfp4_dispatch
from executorch.backends.cuda.triton.kernels import (
    nvfp4_mxfp8_emu as EMU,
    nvfp4_quantized_gemm as GEMM,
    nvfp4_small_m_gemm as SMALL_M,
    nvfp4_w4a8 as W8,
    nvfp4_w4a8_gemv as W4A8_GEMV,
)
from torch.utils._python_dispatch import TorchDispatchMode

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
E2M1 += [-v for v in E2M1]

# Ops that time or run NVFP4 activations (W4A4).
W4A4_OPS = (
    "nvfp4_w4a4",
    "nvfp4_auto_prefill",
    "nvfp4_auto_decode",
    "nvfp4_quantize_activation",
    "nvfp4_w4a4_gate_up",
    "nvfp4_auto_gate_up_prefill",
)


def is_w4a4_op(name, native=True):
    """Whether ``name`` (``op_name``) can time or run NVFP4 activations;
    ``nvfp4_auto_decode`` times W4A4 only with native FP4."""
    if name == "nvfp4_auto_decode":
        return native
    return name in W4A4_OPS or (
        name.startswith("nvfp4_small_m_gemm_m") and not name.endswith(":weight_only")
    )


def op_name(name, args):
    """Small-M bucket ops carry their weight_only flag last."""
    if name.startswith("nvfp4_small_m_gemm_m") and args[-1] is True:
        return f"{name}:weight_only"
    return name


class CleanEnv(unittest.TestCase):
    """Runs every test with the ET_CUDA_NVFP4_* policies at their defaults."""

    def setUp(self):
        env = {
            k: v for k, v in os.environ.items() if not k.startswith("ET_CUDA_NVFP4_")
        }
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)


def native_fp4():
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability()[0] >= 12
    )


def make_weight(n, k, pts=None, seed=0, act_dtype="fp4"):
    """NVFP4-quantize a gaussian weight."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    w = torch.randn(n, k, device="cuda", generator=g) * 0.02
    blocks = w.reshape(n, k // 16, 16)
    amax = blocks.abs().amax(-1)
    if pts is None:
        pts = float(w.abs().max() / (6 * 448))
    pts_t = torch.tensor([pts], device="cuda")
    s = (amax / 6 / pts_t).to(torch.float8_e4m3fn)
    v = blocks / (s.float() * pts_t)[..., None].clamp_min(1e-30)
    grid = torch.tensor(E2M1[:8], device="cuda")
    idx = (v.abs()[..., None] - grid).abs().argmin(-1)
    code = (idx | ((v < 0).long() << 3)).reshape(n, k).to(torch.uint8)
    q = code[:, 0::2] | (code[:, 1::2] << 4)
    return CudaNVFP4Tensor(
        q.contiguous(),
        s.view(torch.uint8).contiguous(),
        pts_t,
        act_dtype=act_dtype,
    )


def stress_weight(n, k, seed=0):
    """Every finite E4M3 scale byte and every E2M1 code, random pairs."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randint(
        0, 256, (n, k // 2), device="cuda", generator=g, dtype=torch.uint8
    )
    s = torch.randint(
        0, 127, (n, k // 16), device="cuda", generator=g, dtype=torch.uint8
    )
    s[:, ::7] = 0
    return CudaNVFP4Tensor(q, s, torch.ones(1, device="cuda"))


def rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _ceil_log2(r):
    bits = r.contiguous().view(torch.int32)
    ex = (bits >> 23) & 255
    e = ex - 127 + ((bits & 8388607) != 0).int()
    e = torch.where(ex == 0, torch.full_like(e, -126), e)
    return e.clamp(-126, 127)


def _pow2(e):
    return ((e.clamp(-126, 127) + 127) << 23).int().view(torch.float32)


def _e4m3(v):
    return v.clamp(-448, 448).to(torch.float8_e4m3fn)


def act_mxfp8_reference(x):
    """Torch mirror of the activation quantizer (``nvfp4_w4a8._act_mxfp8``)."""
    m, k = x.shape
    xf = x.float().reshape(m, k // 32, 32)
    e = _ceil_log2(xf.abs().amax(-1) * torch.tensor(1.0 / 448.0, device=x.device))
    q = _e4m3(xf * _pow2(-e)[..., None])
    deq = (q.float() * _pow2(e)[..., None]).reshape(m, k)
    return q.reshape(m, k).view(torch.uint8), (e + 127).to(torch.uint8), deq


def weight_mxfp8_mirror(qdata, scale):
    """Torch mirror of the in-kernel conversion (``nvfp4_w4a8._decode_fp8_tile``):
    FP16 multiplier and product, then RNE to E4M3."""
    n, half = qdata.shape
    k = half * 2
    lut = torch.tensor(E2M1, device=qdata.device)
    code = lut[torch.stack((qdata & 15, qdata >> 4), -1).reshape(n, k).long()]
    s = scale.view(torch.float8_e4m3fn).float().reshape(n, k // 32, 2)
    e = _ceil_log2(s.amax(-1) * torch.tensor(6.0 / 448.0, device=qdata.device))
    m16 = (s * _pow2(-e)[..., None]).half().float().reshape(n, k // 16)
    prod = (code.reshape(n, k // 16, 16) * m16[..., None]).half().float()
    return (_e4m3(prod).float().reshape(n, k // 32, 32) * _pow2(e)[..., None]).reshape(
        n, k
    )


class OpLog(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.ops = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func.overloadpacket)
        if name.startswith("triton."):
            self.ops.add(op_name(name.split(".")[-1], args))
        return func(*args, **(kwargs or {}))


def linear_ops(act, m, n=512, k=1024, dtype=torch.bfloat16):
    w = make_weight(n, k, act_dtype=act)
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    with OpLog() as log:
        F.linear(x, w)
    return log.ops


class _Linear(torch.nn.Module):
    def __init__(self, act, n=512, k=1024):
        super().__init__()
        self.lin = torch.nn.Linear(k, n, bias=False)
        self.lin.weight = torch.nn.Parameter(
            make_weight(n, k, act_dtype=act), requires_grad=False
        )

    def forward(self, x):
        return self.lin(x)


def exported_ops(act, m, dynamic=False):
    x = torch.randn(m, 1024, device="cuda", dtype=torch.bfloat16)
    shapes = ({0: torch.export.Dim("m", min=17, max=4096)},) if dynamic else None
    ep = torch.export.export(_Linear(act), (x,), dynamic_shapes=shapes, strict=True)
    ep = ep.run_decompositions({})
    return {
        op_name(str(node.target).split(".")[1], node.args)
        for node in ep.graph.nodes
        if node.op == "call_function" and str(node.target).startswith("triton.")
    }


class TestAttribute(CleanEnv):
    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("CUDA required")

    def test_round_trip(self):
        w = make_weight(64, 256, act_dtype="mxfp8")
        names, ctx = w.__tensor_flatten__()
        w2 = CudaNVFP4Tensor.__tensor_unflatten__(
            {n: getattr(w, n) for n in names}, ctx, w.shape, w.stride()
        )
        self.assertEqual(w2.act_dtype, "mxfp8")
        self.assertEqual(w.to(torch.bfloat16).act_dtype, "mxfp8")
        self.assertEqual(w.with_act_dtype("bf16").act_dtype, "bf16")
        self.assertIs(w.with_act_dtype("bf16").qdata, w.qdata)
        fused = CudaNVFP4Tensor.fuse_along_output(
            [w, make_weight(64, 256, act_dtype="mxfp8")]
        )
        self.assertEqual(fused.act_dtype, "mxfp8")
        with self.assertRaises(ValueError):
            CudaNVFP4Tensor.fuse_along_output([w, make_weight(64, 256)])
        with self.assertRaises(ValueError):
            make_weight(64, 256, act_dtype="fp8")

    def test_restored_without_new_defaults_to_fp4(self):
        w = make_weight(64, 256)
        restored = CudaNVFP4Tensor(w.qdata, w.scale, w.per_tensor_scale)
        del restored.act_dtype
        self.assertEqual(restored.act_dtype, "fp4")


class TestDispatchDecisions(CleanEnv):
    """Which kernel each M reaches, with every NVFP4 kernel entry replaced by a
    recorder, for GPUs with and without native FP4 conversion."""

    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("CUDA required")

    def _calls(self, act, m, native, dynamic=False, env=None, k=1024):
        calls = []

        def record(name):
            def fn(x, qdata, *args, **kwargs):
                calls.append(op_name(name, args))
                return x.new_zeros((x.shape[0], qdata.shape[0]))

            return fn

        def family(prefix):
            return lambda bucket: record(f"{prefix}_m{bucket}")

        patches = [
            mock.patch.object(nvfp4_dispatch, "_has_native_fp4", lambda: native),
            mock.patch.object(GEMM, "nvfp4_w4a4", record("nvfp4_w4a4")),
            mock.patch.object(GEMM, "nvfp4_w4a16", record("nvfp4_w4a16")),
            mock.patch.object(GEMM, "nvfp4_gemv", record("nvfp4_gemv")),
            mock.patch.object(
                nvfp4_dispatch, "nvfp4_auto_prefill", record("nvfp4_auto_prefill")
            ),
            mock.patch.object(
                nvfp4_dispatch, "nvfp4_auto_decode", record("nvfp4_auto_decode")
            ),
            mock.patch.object(W4A8_GEMV, "nvfp4_w4a8_gemv", record("nvfp4_w4a8_gemv")),
            mock.patch.object(W8, "nvfp4_w4a8", record("nvfp4_w4a8")),
            mock.patch.object(
                nvfp4_dispatch, "_mxfp8_emulated_linear", record("mxfp8_emulated")
            ),
            mock.patch.object(nvfp4_dispatch, "_dequant_linear", record("dequant")),
            mock.patch.object(
                SMALL_M.NVFP4_SMALL_M_GEMM, "op", family("nvfp4_small_m_gemm")
            ),
            mock.patch.dict(os.environ, env or {}),
        ]
        for p in patches:
            p.start()
        try:
            w = make_weight(512, k, act_dtype=act)
            x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
            if dynamic:
                torch.export.export(
                    _LinearOf(w),
                    (x,),
                    dynamic_shapes=({0: torch.export.Dim("m", min=2, max=4096)},),
                    strict=False,
                )
            else:
                F.linear(x, w)
        finally:
            for p in reversed(patches):
                p.stop()
        return calls

    def test_mxfp8_never_w4a4(self):
        expected = {
            True: {1: "nvfp4_gemv", 64: "nvfp4_w4a8", 4096: "nvfp4_w4a8"},
            # Static M <= 64 keeps the fused W4A16 kernel, like fp4 weights.
            False: {1: "nvfp4_auto_decode", 64: "nvfp4_w4a16", 4096: "dequant"},
        }
        for native in (True, False):
            for m in (1, 2, 3, 4, 5, 8, 9, 16, 17, 64, 4096):
                calls = self._calls("mxfp8", m, native)
                self.assertEqual(len(calls), 1, (native, m, calls))
                self.assertFalse(is_w4a4_op(calls[0], native), (native, m, calls))
                if 2 <= m <= 16:
                    bucket = next(b for b in (2, 4, 8, 16) if m <= b)
                    self.assertEqual(
                        calls[0], f"nvfp4_small_m_gemm_m{bucket}:weight_only"
                    )
                elif m in expected[native]:
                    self.assertEqual(calls[0], expected[native][m], (native, m))
            calls = self._calls("mxfp8", 64, native, dynamic=True)
            self.assertEqual(calls, [expected[native][4096]], native)

    def test_bf16_weight_only(self):
        for native in (True, False):
            for m, kernel in (
                (1, "nvfp4_gemv"),
                (8, "nvfp4_small_m_gemm_m8:weight_only"),
                (64, "nvfp4_w4a16"),
                (4096, "dequant"),
            ):
                self.assertEqual(self._calls("bf16", m, native), [kernel], (native, m))

    def test_policies(self):
        cases = [
            # w4a16 keeps BF16 activations: no MXFP8 prefill, no W4A8 decode.
            ("mxfp8", 64, True, {"ET_CUDA_NVFP4_ACT": "w4a16"}, "nvfp4_w4a16"),
            ("mxfp8", 4096, True, {"ET_CUDA_NVFP4_ACT": "w4a16"}, "dequant"),
            ("mxfp8", 1, False, {"ET_CUDA_NVFP4_ACT": "w4a16"}, "nvfp4_gemv"),
            # w4a8 forces the INT8-activation decode where 8 bits are allowed.
            ("mxfp8", 1, True, {"ET_CUDA_NVFP4_ACT": "w4a8"}, "nvfp4_w4a8_gemv"),
            ("bf16", 1, False, {"ET_CUDA_NVFP4_ACT": "w4a8"}, "nvfp4_gemv"),
            # w4a4 never applies to these weights.
            ("mxfp8", 64, True, {"ET_CUDA_NVFP4_ACT": "w4a4"}, "nvfp4_w4a8"),
            (
                "bf16",
                4096,
                True,
                {"ET_CUDA_NVFP4_PREFILL_IMPL": "kernel"},
                "nvfp4_w4a16",
            ),
            ("mxfp8", 64, True, {"ET_CUDA_NVFP4_PREFILL_IMPL": "dequant"}, "dequant"),
            ("mxfp8", 8, True, {"ET_CUDA_NVFP4_SMALL_M": "0"}, "nvfp4_w4a8"),
            (
                "mxfp8",
                64,
                False,
                {"ET_CUDA_NVFP4_MXFP8_EMULATE": "1"},
                "mxfp8_emulated",
            ),
            # The emulation reproduces SM120 decode too: the W4A16 GEMV.
            ("mxfp8", 1, False, {"ET_CUDA_NVFP4_MXFP8_EMULATE": "1"}, "nvfp4_gemv"),
        ]
        for act, m, native, env, kernel in cases:
            self.assertEqual(self._calls(act, m, native, env=env), [kernel], env)
        with self.assertRaises(ValueError):
            self._calls("mxfp8", 64, False, env={"ET_CUDA_NVFP4_MXFP8_EMULATE": "on"})

    def test_mxfp8_partial_k32_group_dequantizes(self):
        for native in (True, False):
            env = {"ET_CUDA_NVFP4_MXFP8_EMULATE": "1"}
            self.assertEqual(
                self._calls("mxfp8", 64, native, env=env, k=1040), ["nvfp4_w4a16"]
            )
            self.assertEqual(
                self._calls("mxfp8", 4096, native, env=env, k=1040), ["dequant"]
            )

    def test_fp4_unchanged(self):
        self.assertEqual(self._calls("fp4", 64, True), ["nvfp4_auto_prefill"])
        self.assertEqual(self._calls("fp4", 1, True), ["nvfp4_auto_decode"])
        self.assertEqual(self._calls("fp4", 8, True), ["nvfp4_small_m_gemm_m8"])


class _LinearOf(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
        self.lin.weight = torch.nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.lin(x)


def fused_row_scale_weight(k=1024):
    """Two shards with different global scales: per-row per_tensor_scale."""
    return CudaNVFP4Tensor.fuse_along_output(
        [make_weight(256, k, seed=5), make_weight(200, k, seed=6)]
    )


def exact_mxfp8(x, w):
    """MXFP8 activations x the kernel's MXFP8 weights, in FP64."""
    _, _, xq = act_mxfp8_reference(x)
    weight = weight_mxfp8_mirror(w.qdata, w.scale).double()
    return (xq.double() @ weight.t()) * w.per_tensor_scale.double().reshape(1, -1)


def mxfp8_cases():
    """Fused per-row scales, unit scale, FP16, and N/K that are not tile
    multiples (K still a multiple of 32)."""
    unit = make_weight(512, 1024, pts=1.0, seed=7)
    return (
        ("row scales", fused_row_scale_weight(), torch.bfloat16),
        (
            "unit scale",
            CudaNVFP4Tensor(
                unit.qdata, unit.scale, unit.per_tensor_scale, unit_scale=True
            ),
            torch.bfloat16,
        ),
        ("fp16", make_weight(512, 1024, seed=8), torch.float16),
        ("tails", make_weight(500, 1056, seed=9), torch.bfloat16),
    )


class TestNativeCompiles(unittest.TestCase):
    """CI has no SM120 GPU: build the native MXFP8 kernels for sm_120 offline."""

    def test_native_kernels_compile_for_sm120(self):
        from triton.backends.compiler import GPUTarget
        from triton.compiler import ASTSource
        from triton.runtime.jit import JITFunction

        cases = (
            (
                W8._act_mxfp8,
                {"X": "*bf16", "Q": "*u8", "S": "*u8", "M": "i32"},
                {"K": 1024, "BK": 1024},
            ),
            (
                W8._w4a8_mx,
                {
                    "A": "*u8",
                    "AS": "*u8",
                    "B": "*u8",
                    "BS": "*u8",
                    "PTS": "*fp32",
                    "C": "*bf16",
                    "M": "i32",
                },
                {"N": 512, "K": 1024, "BM": 64, "BN": 64, "BK": 128, "GROUP_M": 8},
            ),
        )
        for fn, signature, constants in cases:
            while not isinstance(fn, JITFunction):
                fn = fn.fn
            names = fn.arg_names
            constants = {**constants, "ROW_PTS": True}
            signature = {
                name: "constexpr" if name in constants else signature[name]
                for name in names
                if name in constants or name in signature
            }
            source = ASTSource(
                fn=fn,
                signature=signature,
                constexprs={
                    (names.index(k),): v for k, v in constants.items() if k in names
                },
            )
            try:
                triton.compile(source, target=GPUTarget("cuda", 120, 32))
            except Exception as error:  # noqa: B902
                if "sm_120" in str(error) or "gpu-name" in str(error):
                    self.skipTest(f"toolchain cannot target sm_120: {error}")
                raise


class TestNative(CleanEnv):
    def setUp(self):
        super().setUp()
        if not native_fp4():
            self.skipTest("native FP4 conversion (sm_120) required")

    def _inputs(self, m=256, n=512, k=1024):
        w = make_weight(n, k)
        x = torch.randn(m, k, device="cuda")
        x[:, ::97] *= 8
        x = x.to(torch.bfloat16)
        return w, x, x.float() @ w.dequantize(torch.float32).t()

    def test_op_level_error(self):
        w, x, ref = self._inputs()
        errs = {
            "fp4": rel(
                GEMM.nvfp4_w4a4(x, w.qdata, w.scale, w.per_tensor_scale, None), ref
            ),
            "mxfp8": rel(W8.nvfp4_w4a8(x, w.qdata, w.scale, w.per_tensor_scale), ref),
            "bf16": rel(GEMM.nvfp4_w4a16(x, w.qdata, w.scale, w.per_tensor_scale), ref),
        }
        print("rel error vs BF16 reference:", {a: f"{e:.3e}" for a, e in errs.items()})
        self.assertLess(errs["bf16"], 1e-2)
        self.assertLess(errs["mxfp8"], 5e-2)
        self.assertLess(errs["mxfp8"], errs["fp4"] / 2)
        _, _, xq = act_mxfp8_reference(x)
        exact = (
            xq.double() @ weight_mxfp8_mirror(w.qdata, w.scale).double().t()
        ) * w.per_tensor_scale.double()
        out = W8.nvfp4_w4a8(x, w.qdata, w.scale, w.per_tensor_scale)
        self.assertLess(rel(out, exact), 5e-3)

    def test_mxfp8_cases(self):
        for name, w, dtype in mxfp8_cases():
            x = torch.randn(96, w.shape[1], device="cuda").to(dtype)
            out = F.linear(x, w.with_act_dtype("mxfp8"))
            self.assertEqual(out.dtype, dtype, name)
            self.assertLess(rel(out, exact_mxfp8(x, w)), 5e-3, name)

    def test_activation_quantizer_matches_reference(self):
        torch.manual_seed(1)
        x = torch.randn(64, 1024, device="cuda") * torch.logspace(
            -6, 3, 1024, device="cuda"
        )
        x = x.to(torch.bfloat16)
        q, s = W8.quantize_activation_mxfp8(x)
        rq, rs, _ = act_mxfp8_reference(x)
        self.assertTrue(torch.equal(q, rq))
        self.assertTrue(torch.equal(s, rs))

    def test_in_kernel_weight_conversion(self):
        # One-hot activations and pts = 1: the output is the converted weight.
        n, k = 256, 512
        w = make_weight(n, k, pts=1.0, seed=3)
        x = torch.eye(k, device="cuda", dtype=torch.bfloat16)
        conv, exact = W8.nvfp4_to_mxfp8_weight_reference(w.qdata, w.scale)
        out = W8.nvfp4_w4a8(x, w.qdata, w.scale, w.per_tensor_scale).float()
        self.assertTrue(torch.equal(out.t(), conv))
        nz = exact != 0
        frac = ((conv != exact) & nz).float().sum().item() / nz.sum().item()
        err = ((conv - exact).abs() / exact.abs().clamp_min(1e-30))[nz]
        print(
            f"MXFP8 conversion changed {frac:.4%} of nonzero weights, max rel {err.max():.3e}"
        )
        self.assertLess(err.max().item(), 1 / 16 + 1e-6)
        stress = stress_weight(512, 1024)
        out = W8.nvfp4_w4a8(
            torch.eye(1024, device="cuda", dtype=torch.bfloat16),
            stress.qdata,
            stress.scale,
            stress.per_tensor_scale,
        )
        mirror = weight_mxfp8_mirror(stress.qdata, stress.scale)
        self.assertTrue(torch.equal(out.float().t(), mirror))

    def test_dispatch(self):
        self.assertEqual(linear_ops("mxfp8", 256), {"nvfp4_w4a8"})
        self.assertEqual(linear_ops("mxfp8", 1), {"nvfp4_gemv"})
        self.assertEqual(linear_ops("mxfp8", 8), {"nvfp4_small_m_gemm_m8:weight_only"})
        self.assertEqual(linear_ops("bf16", 256), {"nvfp4_dequant"})

    def test_export_graph(self):
        for m, dynamic, op in (
            (64, True, "nvfp4_w4a8"),
            (1, False, "nvfp4_gemv"),
            (4, False, "nvfp4_small_m_gemm_m4:weight_only"),
        ):
            ops = exported_ops("mxfp8", m, dynamic)
            self.assertIn(op, ops)
            self.assertFalse(any(is_w4a4_op(o) for o in ops), ops)


class TestEmulationKernels(CleanEnv):
    """The emulation kernels on every CUDA GPU (the ARCH >= 89 E4M3 branch
    needs SM89+; CI's A10G/A100 run the software one)."""

    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("CUDA required")

    def test_activation_qdq_bit_identical(self):
        torch.manual_seed(0)
        x = torch.randn(512, 6656, device="cuda") * torch.logspace(
            -6, 3, 6656, device="cuda"
        )
        x[:, ::97] *= 64
        x[3, :64] = 0
        # Block maxima below 448 * 2**-126 share the smallest UE8M0 scale.
        x[4, :32] = 3e-36
        # Block maxima at the UE8M0 boundaries 448 * 2**j and one BF16 ulp
        # around them (446 * 2**j and 450 * 2**j survive the BF16 cast).
        b = torch.tensor([448.0 * 2.0**j for j in range(-20, 10)], device="cuda")
        bits = b.to(torch.bfloat16).view(torch.int16)
        edge = (
            torch.stack([bits, bits + 1, bits - 1], 1).reshape(-1).view(torch.bfloat16)
        )
        x[5, : edge.numel() * 32 : 32] = edge.float()
        x = x.to(torch.bfloat16)
        y, q, s = EMU.act_mxfp8_qdq_codes(x, store_codes=True)
        rq, rs, rdeq = act_mxfp8_reference(x)
        self.assertEqual(int((q != rq).sum()), 0)
        self.assertEqual(int((s != rs).sum()), 0)
        self.assertEqual(int((y.float() != rdeq).sum()), 0)

    def test_weight_conversion_bit_identical(self):
        for name, w in (
            ("pts=1", make_weight(256, 512, pts=1.0, seed=3)),
            ("attention-shaped", make_weight(6656, 4096, seed=1)),
            ("all scale bytes", stress_weight(4096, 4096)),
        ):
            out = EMU.nvfp4_dequant_mxfp8(w.qdata, w.scale).float()
            self.assertEqual(
                int((out != weight_mxfp8_mirror(w.qdata, w.scale)).sum()), 0, name
            )
            if name != "all scale bytes":
                host, _ = W8.nvfp4_to_mxfp8_weight_reference(w.qdata, w.scale)
                self.assertEqual(int((out != host).sum()), 0, name)


class TestEmulated(CleanEnv):
    """``ET_CUDA_NVFP4_MXFP8_EMULATE=1`` dispatch on GPUs without native FP4."""

    def setUp(self):
        super().setUp()
        if not torch.cuda.is_available() or torch.version.hip is not None:
            self.skipTest("CUDA required")
        if native_fp4():
            self.skipTest("native FP4 runs MXFP8 natively")
        env = mock.patch.dict(os.environ, {"ET_CUDA_NVFP4_MXFP8_EMULATE": "1"})
        env.start()
        self.addCleanup(env.stop)

    def test_mxfp8_cases(self):
        for name, w, dtype in mxfp8_cases():
            x = torch.randn(96, w.shape[1], device="cuda").to(dtype)
            out = F.linear(x, w.with_act_dtype("mxfp8"))
            self.assertEqual(out.dtype, dtype, name)
            self.assertLess(rel(out, exact_mxfp8(x, w)), 5e-3, name)

    def test_op_level_error(self):
        w = make_weight(512, 1024)
        x = torch.randn(256, 1024, device="cuda")
        x[:, ::97] *= 8
        x = x.to(torch.bfloat16)
        ref = x.float() @ w.dequantize(torch.float32).t()
        out = F.linear(x, w.with_act_dtype("mxfp8"))
        _, _, xq = act_mxfp8_reference(x)
        exact = (
            xq.double() @ weight_mxfp8_mirror(w.qdata, w.scale).double().t()
        ) * w.per_tensor_scale.double()
        self.assertLess(rel(out, ref), 5e-2)
        self.assertLess(rel(out, exact), 5e-3)
        self.assertLess(rel(F.linear(x, w.with_act_dtype("bf16")), ref), 1e-2)

    def test_dispatch(self):
        self.assertEqual(
            linear_ops("mxfp8", 256), {"nvfp4_mxfp8_act_qdq", "nvfp4_dequant_mxfp8"}
        )
        self.assertEqual(linear_ops("mxfp8", 1), {"nvfp4_gemv"})
        self.assertEqual(linear_ops("mxfp8", 4), {"nvfp4_small_m_gemm_m4:weight_only"})
        self.assertEqual(linear_ops("bf16", 256), {"nvfp4_dequant"})
        self.assertEqual(linear_ops("bf16", 1), {"nvfp4_gemv"})

    def test_export_graph(self):
        for m, dynamic, op in (
            (64, True, "nvfp4_dequant_mxfp8"),
            (1, False, "nvfp4_gemv"),
            (16, False, "nvfp4_small_m_gemm_m16:weight_only"),
        ):
            ops = exported_ops("mxfp8", m, dynamic)
            self.assertIn(op, ops)
            self.assertFalse(any(is_w4a4_op(o, native=False) for o in ops), ops)


if __name__ == "__main__":
    unittest.main()
