"""Wait for an owned diagnostic build and verify its unchanged SDK provenance."""
import hashlib
import json
from pathlib import Path
import time


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        while block:=file.read(16*1024*1024):h.update(block)
    return h.hexdigest()


def qualified_rewind_probe(job,gate_path,original_probe,state,wait_minutes):
    deadline=time.monotonic()+60*wait_minutes
    while True:
        progress=json.loads((job/'progress.json').read_text())
        if progress['status']=='isolated-rewind-probe-ready':break
        assert progress['status']!='failed',progress
        handle=Path('/proc')/str(progress['controller_pid'])/'cmdline'
        assert handle.is_file() and b'prepare_rewind_kv_probe.py' in handle.read_bytes()
        state('waiting-confirmed-live-rewind-probe',probe_build_pid=progress['controller_pid'])
        assert time.monotonic()<deadline,'Rewind probe build observation timeout; inspect same process, do not restart'
        time.sleep(20)
    proof_path=job/'proof.json'
    proof=json.loads(proof_path.read_text())
    gate=json.loads(gate_path.read_text())
    assert proof['native_gate_SHA256']==digest(gate_path)
    assert proof['source']==gate['source'] and proof['original_probe_SHA256']==digest(original_probe)
    assert proof['only_protocol_rewind_and_optional_sampler_added'] and proof['no_inference_kernel_change']
    assert proof['no_original_source_or_binary_changed']
    for name,expected in proof['unchanged_SDK_libraries'].items():assert digest(name)==expected,name
    probe=Path(proof['probe'])
    assert probe.resolve().is_relative_to(job.resolve())
    assert digest(probe)==proof['probe_SHA256']
    assert digest(job/'rewind_kv_probe.cpp')==proof['generated_CPP_SHA256']
    return probe,proof
