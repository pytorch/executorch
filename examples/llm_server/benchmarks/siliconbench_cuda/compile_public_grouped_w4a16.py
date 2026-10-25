"""Compile experimental W4A16 source for explicit targets without CUDA context.

Compilation is source preparation, not numerical or performance acceptance.
No KernelAgent session is invoked and no model dispatch is changed.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import torch
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

import public_grouped_w4a16 as candidate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--targets', nargs='+', type=int, default=[120, 80])
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    assert not torch.cuda.is_initialized()
    report = dict(stage='offline-source-compilation', results=[],
                  source_SHA256=hashlib.sha256(Path(candidate.__file__).read_bytes()).hexdigest(),
                  Triton_version=triton.__version__, PyTorch_version=torch.__version__,
                  numerical_acceptance=False, measured_GPU_performance=False,
                  KA_invoked=False, CUDA_context_initialized=False)
    def save(status):
        report['status'] = status
        temp = a.output / 'progress.pending.json'
        temp.write_text(json.dumps(report, indent=2, default=str) + '\n')
        temp.replace(a.output / 'progress.json')
    save('running')
    try:
        signature = dict(X='*bf16', Q='*u8', SC='*u8', SS='*fp16', Z='*u8',
                         ZS='*fp16', M='i32')
        for target in a.targets:
            # Compile the full SM120 space and a bounded A100 viability subset.
            cfgs = candidate.configs(72) if target == 120 else candidate.configs(72)[:3]
            for index, cfg in enumerate(cfgs):
                cfg.validate()
                for zero_fma in [False, True]:
                    record = dict(target=target, config=asdict(cfg), zero_FMA=zero_fma,
                                  N=39936, K=6656, numerical_acceptance=False)
                    start = time.monotonic()
                    try:
                        sig = {**signature, 'OUT': '*bf16' if cfg.split == 1 else '*fp32'}
                        const = dict(N=39936, K=6656, BM=cfg.bm, BN=cfg.bn, BK=cfg.bk,
                                     SPLIT=cfg.split, GROUP_M=cfg.group_m,
                                     STAGES=cfg.stages, ZERO_FMA=zero_fma)
                        compiled = triton.compile(ASTSource(candidate._gemm, sig, constexprs=const),
                            target=GPUTarget('cuda', target, 32),
                            options=dict(num_warps=cfg.warps, num_stages=cfg.stages,
                                         enable_fp_fusion=False))
                        prefix = f'sm{target}-cfg{index}-fma{int(zero_fma)}'
                        for label in ['ptx', 'ttgir']:
                            raw = compiled.asm[label]
                            (a.output / (prefix + '.' + label)).write_text(raw)
                            record[label + '_SHA256'] = hashlib.sha256(raw.encode()).hexdigest()
                        cubin = compiled.asm['cubin']
                        (a.output / (prefix + '.cubin')).write_bytes(cubin)
                        record['cubin_SHA256'] = hashlib.sha256(cubin).hexdigest()
                        record['metadata'] = compiled.metadata._asdict()
                        record['compiled'] = True
                    except Exception as e:
                        record.update(compiled=False, error_type=type(e).__name__, error=str(e))
                        (a.output / f'sm{target}-cfg{index}-fma{int(zero_fma)}-failure.txt').write_text(traceback.format_exc())
                    record['CPU_compile_elapsed_s'] = time.monotonic() - start
                    report['results'].append(record)
                    assert not torch.cuda.is_initialized(), 'Offline preparation initialized CUDA'
                    save('running')
        report['compiled_count'] = sum(r['compiled'] for r in report['results'])
        report['failed_count'] = len(report['results']) - report['compiled_count']
        save('completed')
    except Exception as error:
        report.update(error_type=type(error).__name__, error=str(error))
        save('failed')
        raise
    print(json.dumps({k:v for k,v in report.items() if k != 'results'}))


if __name__ == '__main__':
    main()
