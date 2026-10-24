"""Compile each explicit-weight GEMM package in a fresh owned process.

Native exports import no candidate modules. Kernel arithmetic, tensor inputs,
dynamic M ranges and launch configurations are preserved. The caller keeps
the benchmark lock, then checks both packages numerically before timing.
"""
import argparse, datetime, hashlib, importlib.util, json, os, subprocess, sys, time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as file:
        while block := file.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def compile_package(inputs, package, *, et_source, bucket, config_id,
                    baseline, label, reference_kind, deadline_utc,
                    precision_control=None):
    import torch
    package = Path(package)
    folder = package.parent / 'isolated-export' / package.stem
    folder.mkdir(parents=True, exist_ok=False)
    tensor_file = folder / 'explicit-inputs.pt'
    torch.save(tuple(t.detach().cpu() for t in inputs), tensor_file)
    request = dict(parent_pid=os.getpid(), et_source=str(et_source), bucket=bucket,
                   config_id=config_id, baseline=baseline, label=label,
                   reference_kind=reference_kind, precision_control=str(precision_control) if precision_control else None,
                   deadline_utc=deadline_utc, tensor_file=str(tensor_file),
                   tensor_file_sha256=digest(tensor_file), package=str(package),
                   folder=str(folder))
    path = folder / 'request.json'
    path.write_text(json.dumps(request, indent=2) + '\n')
    remaining = (datetime.datetime.fromisoformat(deadline_utc.replace('Z', '+00:00')) - datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    assert remaining > 0
    with (folder / 'export.log').open('x') as log:
        child = subprocess.Popen([sys.executable, '-B', str(Path(__file__).resolve()), '--request', str(path)],
                                 stdout=log, stderr=subprocess.STDOUT)
        try:
            assert child.wait(timeout=min(900, remaining)) == 0, 'Inspect preserved isolated export.log'
        finally:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=10)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
    end = time.monotonic() + 60
    while str(child.pid) in subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).splitlines():
        assert time.monotonic() < end, 'Export child CUDA context remains live'
        time.sleep(1)
    result = json.loads((folder / 'export-metadata.json').read_text())
    assert result['package_sha256'] == digest(package)
    assert result['input_tensor_file_sha256'] == request['tensor_file_sha256']
    assert result['fresh_export_process_pid'] != os.getpid()
    if label == 'reference': assert not result['candidate_modules_imported']
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    r = json.loads(args.request.read_text())
    assert r['parent_pid'] == os.getppid()
    folder = Path(r['folder'])
    assert args.request.resolve().parent == folder.resolve()
    assert digest(r['tensor_file']) == r['tensor_file_sha256']
    assert datetime.datetime.now(datetime.timezone.utc) < datetime.datetime.fromisoformat(r['deadline_utc'].replace('Z', '+00:00'))
    assert r['bucket'] in [8, 16] and r['label'] in ['reference', 'candidate']
    os.environ.update(TRITON_CACHE_DIR=str(folder / 'triton-cache'),
                      TORCHINDUCTOR_CACHE_DIR=str(folder / 'inductor-cache'),
                      TORCHINDUCTOR_COMPILE_THREADS='1', MAX_JOBS='1', OMP_NUM_THREADS='2')
    sys.path.insert(0, str(Path(r['et_source']).parent))
    import torch
    torch.set_num_threads(2)
    assert torch.cuda.get_device_capability() == (12, 0)
    inputs = tuple(t.cuda() for t in torch.load(r['tensor_file'], weights_only=True))
    assert len(inputs) == 6 and inputs[0].shape[0] == r['bucket']
    if r['label'] == 'reference':
        import triton
        from executorch.backends.cuda.triton.kernels import int4_quantized_gemm as native
        from executorch.backends.cuda.triton.kernels import quantized_gemm_family as family
        assert digest(native.__file__) == 'fb72d31e6d3f91e076fe9721744eacb7c379ffe0612c57e3c48ae4a25c20027c'
        launch_module = native
        if r['reference_kind'] == 'Gemma-precision-native':
            assert digest(r['precision_control']) == '10affaf3cd36f17a68f8336e764e24b16e54dd509d89577c9009445b2f41cf9b'
            spec = importlib.util.spec_from_file_location('isolated_unchanged_precision_reference', r['precision_control'])
            launch_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(launch_module)
        else:
            assert r['reference_kind'] == 'MG-native'
        c = r['baseline']['config']
        kwargs = c.get('kwargs', {k: v for k, v in c.items() if k not in ['num_warps', 'num_stages', 'num_ctas']})
        cfg = triton.Config(kwargs, num_warps=c['num_warps'], num_stages=c['num_stages'], num_ctas=c.get('num_ctas', 1))
        def launch(bucket, x, q, sc, ss, z, zs, group_size):
            assert bucket == r['bucket'] and group_size == 32
            return launch_module._launch_w4a16(bucket, x, q, sc, ss, z, zs, 32,
                                                split_k=r['baseline']['split_k'], config=cfg)
        registered = family.QuantizedGemmFamily('isolated_native_' + str(r['bucket']), [r['bucket']],
                                               launch_module._prototype, launch, launch_module._fake, launch_module._unsupported_reason)
        op = registered.op(r['bucket'])
        class Module(torch.nn.Module):
            def forward(self, x, q, sc, ss, z, zs):
                return op(x, q, sc, ss, z, zs, 32)
    else:
        from candidates_w4a16_v9.aoti import candidate_op, config_index
        index = config_index(r['config_id'])
        class Module(torch.nn.Module):
            def forward(self, x, q, sc, ss, z, zs):
                return candidate_op(x, q, sc, ss, z, zs, index)
    lower = 5 if r['bucket'] == 8 else 9
    dynamic = ({0: torch.export.Dim('active_M', min=lower, max=r['bucket'])}, None, None, None, None, None)
    with torch.inference_mode(), torch.cuda.stream(torch.cuda.Stream()):
        ep = torch.export.export(Module(), inputs, dynamic_shapes=dynamic, strict=True)
        torch._inductor.aoti_compile_and_package(ep, package_path=r['package'])
    candidates = [name for name in sys.modules if name.startswith('candidates_w4a16')]
    if r['label'] == 'reference': assert not candidates
    report = dict(package_sha256=digest(r['package']), package_bytes=Path(r['package']).stat().st_size,
                  export_range_constraints=str(ep.range_constraints), dynamic_M_min=lower, dynamic_M_max=r['bucket'],
                  fresh_export_process_pid=os.getpid(), candidate_modules_imported=candidates,
                  input_tensor_file_sha256=r['tensor_file_sha256'], native_arithmetic_and_config_unchanged=True,
                  numerical_acceptance_pending=True)
    (folder / 'export-metadata.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
