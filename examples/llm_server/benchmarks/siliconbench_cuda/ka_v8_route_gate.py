"""Prepared acceptance loader for a fresh isolated full-model export.

Read the complete actual-GGUF/AOTI report and accept only material current
winning routes. This file is not published, deployed or invoked by a server.
"""
from pathlib import Path
import hashlib
import json
import math
import statistics


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        while chunk := source.read(8*1024*1024):
            h.update(chunk)
    return h.hexdigest()


def validated_routes(path, candidate_source):
    path = Path(path)
    source = Path(candidate_source)
    state = json.loads((path.parent/'progress.json').read_text())
    assert state['status'] == 'completed'
    owner = json.loads((path.parent/'lock-owner.json').read_text())['pid']
    for pid in [owner, state.get('controller_pid')]:
        if pid is None:
            continue
        process = Path('/proc')/str(pid)/'stat'
        assert not process.exists() or process.read_text().rsplit(')',1)[1].split()[0] in ['Z','X']
    report = json.loads(path.read_text())
    assert report['rounds']==4 and report['discarded']==[1] and report['averaged']==[2,3,4]
    assert report['warmups']==3 and report['measured_buckets']==[8,16]
    assert report['GGUF_sha256']=='7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a'
    release = Path(report['public_source_release'])
    manifest = json.loads((release/'public-source-manifest.json').read_text())
    assert digest(release/'public-source-manifest.json')==report['public_source_manifest_sha256']
    for name,proof in manifest['files'].items():
        assert digest(release/name)==proof['sha256']
        if name.startswith('candidates_w4a16_v8/'):
            assert digest(source/name)==proof['sha256'], 'Candidate source changed after the measured gate'

    def numeric(check):
        assert check['deterministic'] and math.isfinite(check['relative_rms']) and check['relative_rms']<.005

    def timing(t):
        assert t['orders']==[['reference','candidate'],['candidate','reference'],['reference','candidate'],['candidate','reference']]
        for role in ['reference','candidate']:
            samples=t['samples_us'][role]
            assert len(samples)==4 and all(math.isfinite(x) and x>0 for x in samples)
            assert math.isclose(t['means_us'][role],statistics.mean(samples[1:]),rel_tol=1e-9)
            assert math.isclose(t['sample_sd_us'][role],statistics.stdev(samples[1:]),rel_tol=1e-9)
        assert math.isclose(t['candidate_over_reference'],t['means_us']['candidate']/t['means_us']['reference'],rel_tol=1e-9)

    assert len(report['cases'])==12
    real={(c['role'],c['m']) for c in report['cases'] if c['role'].startswith('Gemma-real-')}
    assert real=={(role,m) for role in ['Gemma-real-attn_k','Gemma-real-ffn_down','Gemma-real-fused_gate_up','Gemma-real-vocabulary'] for m in [8,16]}
    assert {(c['m'],c['n'],c['k']) for c in report['cases'] if c['role']=='MG-synthetic'}=={(m,n,6656) for m in [8,16] for n in [19968,39936]}
    pairs=0
    for case in report['cases']:
        if case['role'].startswith('Gemma-real-'):
            assert case['reference_numerical_gate_required'];numeric(case['reference_check'])
        for candidate in case['candidates']:
            assert candidate['config']['bucket']==case['m']
            numeric(candidate['numerics']);timing(candidate['paired_timing']);pairs+=1
    assert pairs==30
    expected={('Gemma-real-attn_k',8),('Gemma-real-attn_k',16),('Gemma-real-ffn_down',8),('Gemma-real-fused_gate_up',16)}
    assert len(report['AOTI_cases'])==4 and {(c['role'],c['bucket']) for c in report['AOTI_cases']}==expected
    routes={};decisions=[]
    for case in report['AOTI_cases']:
        b=case['bucket'];lower=5 if b==8 else 9
        assert case['dynamic_M_min']==lower and case['dynamic_M_max']==b
        assert case['AOTI_timing_policy']=='unchanged native cuda_graph_timing.time_with_cuda_graph_us; cold L2 per call; 4-32 calls in one graph; minimum GPU event interval'
        assert case['AOTI_invocation_policy']=='official AOTInductorModelContainerRunSingleThreaded with explicit current CUDA stream'
        assert case['selected_policy']['use_candidate'] and not case['selected_policy']['pending_synthetic']
        assert set(case['packages'])=={'reference','candidate'}
        for package in case['packages'].values():
            p=Path(package['path']);assert p.resolve().is_relative_to((path.parent/'aoti-packages').resolve())
            assert p.stat().st_size==package['bytes'] and digest(p)==package['sha256']
        assert len(case['tail_numerics'])==3 and {v['m'] for v in case['tail_numerics']}=={lower,b-1,b}
        for tail in case['tail_numerics']:
            assert set(tail['checks'])=={'reference','candidate'}
            for check in tail['checks'].values():numeric(check)
        t=case['paired_timing'];timing(t)
        use=t['candidate_over_reference']<.98
        key=(b,case['n'],case['k'])
        if use:routes[key]=case['config_id']
        decisions.append(dict(shape=list(key),config=case['config_id'],candidate_over_reference=t['candidate_over_reference'],use_candidate=use))
    assert routes, 'No material current AOTI gain; keep the native artifact'
    return routes, dict(source_report_sha256=digest(path),source_release_manifest_sha256=report['public_source_manifest_sha256'],
                        real_eager_cases=8,eager_pairs=30,AOTI_pairs=4,tail_checks=24,route_decisions=decisions,
                        whole_model_performance_memory_F1_acceptance_pending=True)
