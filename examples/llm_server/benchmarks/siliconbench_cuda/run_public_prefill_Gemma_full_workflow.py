"""Qualify a full Gemma A16 prefill candidate with the primary server protocol.

Retain the selected primary artifact, native runtimes and every old result.
Wait for the bounded Down routing correction, export all 48 original layers,
then collect the complete performance grid and the agreed F1 coverage.
The script never promotes a candidate automatically.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
import os


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(16 * 1024 * 1024):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['down-AOTI-job', 'gate-AOTI-report', 'reference-artifact',
                 'reference-export-job', 'et-source', 'gguf', 'config',
                 'precision-control', 'base-wrapper', 'client', 'prompts',
                 'correctness-root', 'lock', 'job-dir']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--benchmark-python', required=True)
    p.add_argument('--port', type=int, default=19111)
    a = p.parse_args()
    assert not a.job_dir.exists(), 'Retain every earlier attempt'
    a.job_dir.mkdir(parents=True)
    package = Path(__file__).resolve().parent

    def state(status, **fields):
        pending = a.job_dir / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            full_model=True, model_promoted=False, new_KA=False, **fields), indent=2)+'\n')
        pending.replace(a.job_dir / 'progress.json')

    def run(command, label, timeout):
        with (a.job_dir / (label+'.log')).open('x') as log:
            child = subprocess.Popen([str(v) for v in command], stdout=log, stderr=subprocess.STDOUT)
            state('running-'+label, child_pid=child.pid)
            try:
                assert child.wait(timeout=timeout) == 0, label
            except BaseException:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
                raise

    try:
        state('waiting-existing-Down-AOTI-control')
        deadline = time.monotonic()+7200
        while True:
            progress = json.loads((a.down_AOTI_job/'progress.json').read_text())
            if progress['status'] == 'completed':
                break
            assert progress['status'] != 'failed', progress
            os.kill(progress['controller_pid'], 0)
            assert time.monotonic() < deadline, 'Observation expired; do not restart the AOTI control'
            time.sleep(20)
        down_report = a.down_AOTI_job/'results.json'
        reference = json.loads((a.reference_artifact/'artifact-audit.json').read_text())
        assert reference['constant_metadata']['get_n_caches'] == 48
        assert reference['physical_columns'] == 19 and reference['max_resident_sequences'] == 18
        assert reference['context'] == 131072 and reference['safe_ring_rows'] == 1535
        assert reference['selected_output_bound'] == [64,262144]
        assert reference['static_methods'] == [1,2,4,8,16,32,64]
        for name, item in reference['artifacts'].items():
            assert sha(a.reference_artifact/name) == item['sha256']
        assert sha(a.base_wrapper) == '84aa8d997c6c6981a2cc4fc541c264f6bd8f42a3b1642af15bdae32fc1be0020'
        assert sha(a.client) == 'd2f9eba743b583b9b5968d2f8704cb347638a05fe12717797d21a8c938a00b05'
        assert sha(a.prompts) == '76d3ff55f4026a50679fd60becd07a9cad93549dc46d98a59bcd642c90d0f81e'
        assert sha(a.precision_control) == '10affaf3cd36f17a68f8336e764e24b16e54dd509d89577c9009445b2f41cf9b'
        assert sha(a.correctness_root.parent/'GMRID-input-audit.json') == 'c94ff150d2352e91c9e29d7ace47a75e883041ffc071d7aa2b57aac813d0ae1c'
        cache = a.reference_export_job/'cache'
        cache_bytes = sum(f.stat().st_size for name in ['inductor','triton']
                          for f in (cache/name).rglob('*') if f.is_file())
        required = 3*reference['artifacts']['aoti_cuda_blob.ptd']['bytes']+cache_bytes+4*1024**3
        available = shutil.disk_usage(a.job_dir).free
        (a.job_dir/'export-storage-preflight.json').write_text(json.dumps(dict(
            available_bytes=available,required_bytes=required,temporary_blob_copies_budgeted=3),indent=2)+'\n')
        assert available > required, 'Insufficient export staging space'
        output = a.job_dir/'public-full-artifact'
        run([sys.executable,'-B',package/'export_public_prefill_Gemma_full.py',
            '--et-source',a.et_source,'--gguf',a.gguf,'--config',a.config,
            '--precision-control',a.precision_control,'--max-step-tokens','512',
            '--max-kv-seqs','18','--max-context','131072',
            '--reference-artifact',a.reference_artifact,'--reference-export-job',a.reference_export_job,
            '--gate-AOTI-report',a.gate_AOTI_report,'--down-AOTI-report',down_report,
            '--output',output,'--job-dir',a.job_dir/'full-export','--lock',a.lock],
            'full-model-export',22800)
        audit = json.loads((output/'artifact-audit.json').read_text())
        assert audit['full_weight_PTD_bytes_identical_with_primary']
        assert audit['constant_metadata'] == reference['constant_metadata']
        assert sum(x['candidate_operator_calls'] for x in audit['actual_strict_export_route_audit']) == 96
        body = a.base_wrapper.read_text()
        pattern = re.compile(r'(?m)^([ \t]*)artifact=([^\n]+)$')
        matches = list(pattern.finditer(body))
        assert len(matches) == 1 and str(a.reference_artifact) in matches[0].group(2)
        body = pattern.sub(lambda m: m[1]+'artifact='+shlex.quote(str(output.resolve())),body)
        assert 'et_effective_checkpoint=512' in body
        assert '--initial-kv-capacity 512' in body
        wrapper = a.job_dir/'serve-candidate.sh'
        wrapper.write_text(body)
        subprocess.run(['bash','-n',str(wrapper)],check=True)
        (a.job_dir/'server-protocol.json').write_text(json.dumps(dict(
            original_wrapper=str(a.base_wrapper), original_wrapper_SHA256=sha(a.base_wrapper),
            candidate_wrapper_SHA256=sha(wrapper), changes=['model artifact path'],
            client_SHA256=sha(a.client), prompts_SHA256=sha(a.prompts),
            prefix_entries=1, checkpoint_by_prefix={'off':0,'on':512},
            context=131072,initial_KV=512,active_sessions=16,decode_ceiling=16,
            max_prefill_tokens=512,static_graphs=True,weight_PTD_bytes_identical=True,
            correctness_request_overrides=None,full_model_F1_pending=True),indent=2)+'\n')
        command = [a.benchmark_python,'-B',package/'run_matrix_24h.py',
            '--wrapper',wrapper,'--model','gemma4-12b-it','--client',a.client,
            '--prompts',a.prompts,'--correctness-root',a.correctness_root,
            '--python',a.benchmark_python,'--lock',a.lock,'--port',a.port,
            '--variants','executorch','--cache-modes','off','on',
            '--concurrencies','16','8','4','2','1']
        run(command+['--output',a.job_dir/'server-performance','--phases','performance'],
            'full-model-server-performance',21600)
        run(command+['--output',a.job_dir/'server-correctness','--phases','correctness'],
            'full-model-server-F1',57600)
        state('completed-pending-independent-full-model-selection',
              server_perf_memory_and_F1_collected=True)
    except BaseException as error:
        state('failed',error_type=type(error).__name__,detail=str(error)[-2000:])
        raise


if __name__ == '__main__':
    main()
