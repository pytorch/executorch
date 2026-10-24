# Optimized by KernelAgent-Oink(https://github.com/meta-pytorch/KernelAgent)
"""Independent synthetic validation for the bounded planar-Q6 W6A16 family."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from . import CONFIGS, run
from . import common


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pack_q6(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack explicit signed Q6 values without using candidate indexing."""
    n, k = q.shape
    u = (q + 32).to(torch.uint8)
    low = u & 15
    ql = (low[:, 0::2] | (low[:, 1::2] << 4)).contiguous()
    # Native Q6_K stores eight K32 chunks per K256. Within each chunk,
    # bytes are [parity, octet], and each byte packs four 2-bit fields.
    fields = (u >> 4).reshape(n, k // 256, 8, 4, 4, 2)
    fields = fields.permute(0, 1, 2, 5, 3, 4)  # chunk, parity, octet, field
    shifts = torch.arange(0, 8, 2, dtype=torch.uint8, device=q.device)
    qh = torch.sum(fields << shifts, dim=-1).to(torch.uint8)
    return ql, qh.reshape(n, k // 4).contiguous()


def _independent_q(ql: torch.Tensor, qh: torch.Tensor) -> torch.Tensor:
    """Decode by reshaping planes; this is structurally unlike kernel address math."""
    n, k2 = ql.shape
    sb = k2 // 128
    low_bytes = ql.reshape(n, sb, 128)
    low = torch.stack((low_bytes & 15, low_bytes >> 4), dim=-1).reshape(n, sb, 256)
    high_bytes = qh.reshape(n, sb, 8, 2, 4)  # chunk, parity, octet
    fields = torch.stack(tuple((high_bytes >> shift) & 3 for shift in (0, 2, 4, 6)), dim=-1)
    high = fields.permute(0, 1, 2, 4, 5, 3).reshape(n, sb, 256)  # octet, field, parity
    return (low | (high << 4)).to(torch.int16).reshape(n, sb * 256) - 32


def _plane_layout_check() -> dict:
    k = 512
    base = torch.arange(k, dtype=torch.int16)
    patterns = torch.stack(((base % 64) - 32, ((base // 32) % 64) - 32, ((base * 29 + 7) % 64) - 32))
    ql, qh = _pack_q6(patterns)
    got = _independent_q(ql, qh)
    torch.testing.assert_close(got, patterns, rtol=0, atol=0)
    return {"name": "independent-plane-layout", "status": "pass", "covers": ["K32", "K128", "K256", "K512"]}


def _make_problem(m: int, n: int, k: int, seed: int):
    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(seed)
    q = torch.randint(-32, 32, (n, k), dtype=torch.int16, device=device, generator=gen)
    ql, qh = _pack_q6(q)
    del q
    scale = torch.randint(-31, 32, (n, k // 16), dtype=torch.int8, device=device, generator=gen)
    scale.view(-1)[:6] = torch.tensor([-128, -127, -1, 0, 1, 127], dtype=torch.int8, device=device)
    choices = torch.tensor([0.0, 0.03125, -0.0625, 0.2, 0.75, -1.25], dtype=torch.float16, device=device)
    steps = choices[torch.randint(0, len(choices), (n, k // 256), device=device, generator=gen)].contiguous()
    x = (torch.randn((m, k), dtype=torch.float32, device=device, generator=gen) * 0.35).to(torch.bfloat16)
    if seed & 1:
        x[0].zero_()
    return x.contiguous(), ql, qh, scale.contiguous(), steps


def _oracle(x, ql, qh, scale, steps, chunk_n: int = 64) -> torch.Tensor:
    pieces = []
    for first in range(0, ql.shape[0], chunk_n):
        last = min(first + chunk_n, ql.shape[0])
        q = _independent_q(ql[first:last], qh[first:last]).to(torch.bfloat16)
        code = scale[first:last].to(torch.float32).to(torch.bfloat16)
        step = steps[first:last].to(torch.bfloat16)
        group_scale = (code.reshape(last - first, -1, 16) * step[:, :, None]).to(torch.bfloat16).reshape(last - first, -1)
        weight = (q * group_scale.repeat_interleave(16, dim=1)).to(torch.bfloat16)
        pieces.append((x.float() @ weight.float().T).to(torch.bfloat16))
    return torch.cat(pieces, dim=1)


def _check(got: torch.Tensor, ref: torch.Tensor) -> dict:
    if got.dtype != torch.bfloat16 or not got.is_contiguous() or got.shape != ref.shape:
        raise AssertionError("output contract mismatch")
    gf, rf = got.float(), ref.float()
    torch.testing.assert_close(gf, rf, rtol=0.03, atol=3.0)
    diff = gf - rf
    rms = float(torch.sqrt(torch.mean(diff * diff)))
    ref_rms = float(torch.sqrt(torch.mean(rf * rf)))
    if rms > 0.02 * max(ref_rms, 1.0):
        raise AssertionError(f"RMS error {rms} exceeds normalized bound; ref RMS {ref_rms}")
    return {"max_abs": float(diff.abs().max()), "rms": rms, "ref_rms": ref_rms}


def _record_case(results: list, name: str, bucket: int, config_id: str, tensors, ref=None) -> None:
    try:
        x, ql, qh, scale, steps = tensors
        if ref is None:
            ref = _oracle(x, ql, qh, scale, steps)
        got = run(x, ql, qh, scale, steps, bucket=bucket, config=config_id)
        metrics = _check(got, ref)
        results.append({"name": name, "config": config_id, "status": "pass", **metrics})
    except Exception as error:  # preserve the complete validation matrix
        results.append({"name": name, "config": config_id, "status": "fail", "error": repr(error)})


def _contract_checks(tensors) -> list[dict]:
    x, ql, qh, scale, steps = tensors
    checks = []
    probes = (
        ("unknown-config", lambda: run(x, ql, qh, scale, steps, bucket=8, config="not-declared")),
        ("bucket-config-mismatch", lambda: run(x, ql, qh, scale, steps, bucket=16, config="w6v6_m8_k256_n32_w2_s1_sp1")),
        ("modified-config", lambda: run(x, ql, qh, scale, steps, bucket=8, config={**CONFIGS["w6v6_m8_k256_n32_w2_s1_sp1"], "split_k": 4})),
        ("split32-reject-k6656", lambda: common.validate_split(32, 6656)),
    )
    for name, probe in probes:
        try:
            probe()
        except ValueError:
            checks.append({"name": name, "status": "pass"})
        else:
            checks.append({"name": name, "status": "fail", "error": "expected ValueError"})
    try:
        common.validate_split(32, 19968)
        checks.append({"name": "split32-accept-k19968", "status": "pass"})
    except Exception as error:
        checks.append({"name": "split32-accept-k19968", "status": "fail", "error": repr(error)})
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quick", action="store_true")
    mode.add_argument("--all", action="store_true")
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    root = Path(__file__).resolve().parent
    results = [_plane_layout_check()]
    small = (
        ("direct-m1-ntail", 1, 33, 256, "w6v6_m8_k256_n32_w2_s1_sp1"),
        ("direct-m8-ntail", 8, 33, 256, "w6v6_m8_k256_n32_w2_s1_sp1"),
        ("split-m7-ntail", 7, 65, 512, "w6v6_m8_k256_n64_w2_s1_sp2"),
        ("k128-m16", 16, 31, 256, "w6v6_m16_k128_n64_w4_s2_sp1"),
        ("split8-m15-ntail", 15, 67, 2048, "w6v6_m16_k128_n64_w4_s2_sp8"),
    )
    first_tensors = None
    for index, (name, m, n, k, cid) in enumerate(small):
        tensors = _make_problem(m, n, k, 100 + index)
        first_tensors = first_tensors or tensors
        _record_case(results, name, CONFIGS[cid]["bucket"], cid, tensors)
    results.extend(_contract_checks(first_tensors))

    # Explicit packed-plane extremes exercise candidate decode, not only the
    # CPU pack/decode round-trip above.
    for value, label in ((0, "all-zero"), (255, "all-ff")):
        tensors = list(_make_problem(8, 35, 256, 700 + value))
        tensors[1].fill_(value)
        tensors[2].fill_(value)
        _record_case(results, f"packed-{label}-ql-qh", 8,
                     "w6v6_m8_k256_n32_w2_s1_sp1", tuple(tensors))

    if args.all:
        roles = (("ffn_down", 6656, 19968), ("attn_v", 256, 6656))
        for bucket in (8, 16):
            ids = [cid for cid, cfg in CONFIGS.items() if cfg["bucket"] == bucket]
            for role_index, (role, n, k) in enumerate(roles):
                tensors = _make_problem(bucket, n, k, 1000 + bucket * 10 + role_index)
                ref = _oracle(*tensors)
                for cid in ids:
                    _record_case(results, f"{role}-M{bucket}", bucket, cid, tensors, ref)
                del tensors, ref
                torch.cuda.empty_cache()

        cid = "w6v6_m8_k256_n32_w2_s1_sp8"
        tensors = _make_problem(8, 65, 2048, 9090)
        try:
            a = run(*tensors, bucket=8, config=cid)
            b = run(*tensors, bucket=8, config=cid)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            results.append({"name": "determinism", "config": cid, "status": "pass"})
        except Exception as error:
            results.append({"name": "determinism", "config": cid, "status": "fail", "error": repr(error)})

    failures = [result for result in results if result["status"] != "pass"]
    report = {
        "schema": "w6a16-v6-validation-v1",
        "mode": "all" if args.all else "quick",
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "config_count": len(CONFIGS),
        "case_count": len(results),
        "failure_count": len(failures),
        "source_sha256": {path.name: _sha(path) for path in root.glob("*.py")},
        "results": results,
    }
    output = Path(args.json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"report": str(output), "cases": len(results), "failures": len(failures)}))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
