"""Experimental larger-M buckets using the existing ET W4A16 Triton bodies.

No new arithmetic, packing, activation quantization or reduction is introduced.
Existing 1..64 tuners are retained. This is a process-local candidate hook;
serving integration and a crossover policy require separate measured acceptance.
"""
import hashlib
from pathlib import Path


BUCKETS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384)


def configs(bucket):
    import triton
    from executorch.backends.cuda.triton.kernels import quantized_gemm_utils as u
    return [triton.Config(dict(BLOCK_M=bm, BLOCK_N=bn, PIPELINE_STAGES=stages),
                          num_warps=warps, num_stages=stages)
            for bm in (*u.TILE_BLOCK_M_CHOICES, 128)
            if bm <= bucket
            for bn in u.TILE_BLOCK_N_CHOICES
            for warps in u.TILE_WARP_CHOICES
            for stages in u.PIPELINE_STAGE_CHOICES]


def install(*, serving_dispatch=False, crossover=None):
    import triton
    from executorch.backends.cuda.triton.kernels import int4_quantized_gemm as native
    from executorch.backends.cuda.triton.kernels.quantized_gemm_family import QuantizedGemmFamily
    assert hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest() == \
        'fb72d31e6d3f91e076fe9721744eacb7c379ffe0612c57e3c48ae4a25c20027c'
    assert not any(b in native._W4A16_DIRECT_KERNELS for b in BUCKETS)
    for bucket in BUCKETS:
        for target, body in [(native._W4A16_DIRECT_KERNELS, native._int4_w4a16_direct_kernel),
                             (native._W4A16_SPLIT_KERNELS, native._int4_w4a16_splitk_kernel)]:
            target[bucket] = triton.autotune(
                configs=configs(bucket), key=['M', 'N', 'K', 'SPLIT_K'],
                prune_configs_by={'early_config_prune': native._prune_tiles},
                do_bench=native.autotune_do_bench)(body)
    if serving_dispatch:
        assert crossover in BUCKETS, 'Choose only after full-forward comparison'
        from executorch.backends.cuda.quantize_op_dispatch import _gemm_family_dispatch as dispatch
        native.INT4_QUANTIZED_GEMM = QuantizedGemmFamily(
            'roofline_large_int4_quantized_gemm',
            (*native.SUPPORTED_BUCKETS, *(b for b in BUCKETS if b <= crossover)),
            native._prototype, native._launch, native._fake, native._unsupported_reason)
        dispatch.DEFAULT_MAX_BUCKET = crossover
    return dict(buckets=list(BUCKETS), configs_per_bucket={b: len(configs(b)) for b in BUCKETS},
                native_bodies_unchanged=True, old_tuners_unchanged=True,
                candidate_serving_dispatch=serving_dispatch, crossover=crossover)
