import json

import torch

from executorch.backends.cuda.dp4a_planar_int6_tensor import unpack_int6
from executorch.backends.cuda.quantize_op_dispatch.int6_dispatch import _unit_dq_mm_int6
from executorch.backends.cuda.tests.test_int6_large_quantized_gemm import (
    GROUP_SIZE,
    TILE_SPLIT_K_CANDIDATES,
    _sampled,
    kernels,
    legal_configs,
    packed_weights,
)


def errors(actual, expected):
    actual, expected = actual.float().cpu(), expected.float().cpu()
    delta = (actual - expected).abs()
    return {
        "mismatches": int((delta > 4.0 + 0.02 * expected.abs()).sum()),
        "max_abs": float(delta.max()),
        "mean_rel": float(delta.mean() / expected.abs().mean().clamp_min(1e-6)),
    }


failures = 0
for bucket in (8, 16, 32, 64):
    for n in (37, 257):
        for k in (256, 5376):
            weights = packed_weights(n, k, seed=n ^ k)
            ql, qh, scale, steps = (w.cpu() for w in weights)
            q = unpack_int6(ql, qh, n, k).to(torch.bfloat16).reshape(n, k // GROUP_SIZE, GROUP_SIZE)
            s = scale.to(torch.bfloat16) * steps.to(torch.bfloat16).repeat_interleave(16, dim=1)
            dequant = (q * s[..., None]).reshape(n, k).float()
            widest = max(c for c in TILE_SPLIT_K_CANDIDATES if c <= k // 256)
            for m in (bucket, bucket - 1):
                x = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
                ref = _unit_dq_mm_int6(x, *weights, GROUP_SIZE)
                oracle = x.float().cpu() @ dequant.T
                print("REFERENCE", json.dumps({"m": m, "n": n, "k": k, **errors(ref, oracle)}), flush=True)
                for split_k in sorted({1, widest}):
                    for config in _sampled(legal_configs(bucket, k, split_k)):
                        out = kernels._launch_w6a16(bucket, x, *weights, GROUP_SIZE, split_k=split_k, config=config)
                        torch.cuda.synchronize()
                        error = errors(out, ref)
                        bad = error["mismatches"] > 0 or error["mean_rel"] >= 0.01
                        failures += bad
                        print("CONFIG", json.dumps({"m": m, "n": n, "k": k, "bucket": bucket, "split_k": split_k, "config": str(config), "error": error, "oracle_error": errors(out, oracle), "failed": bad}), flush=True)
                        if bad:
                            print("SAMPLES", "out", out.flatten()[:16].tolist(), "ref", ref.flatten()[:16].tolist(), "oracle", oracle.flatten()[:16].tolist(), flush=True)
            del weights
            torch.cuda.empty_cache()
print("INT6_FAILURE_COUNT", failures, flush=True)
assert failures == 0
