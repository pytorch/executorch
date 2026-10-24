"""Prepared gate for MG routes; the complete real-GGUF report is mandatory."""
from pathlib import Path
import hashlib,json,math,statistics


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(8*1024*1024):h.update(chunk)
    return h.hexdigest()


def validated_routes(path,candidate_source):
    path=Path(path);source=Path(candidate_source)
    state=json.loads((path.parent/'progress.json').read_text());assert state['status']=='completed'
    owner=json.loads((path.parent/'lock-owner.json').read_text())['pid']
    for pid in [owner,state.get('controller_pid')]:
        if pid is None:continue
        p=Path('/proc')/str(pid)/'stat'
        assert not p.exists() or p.read_text().rsplit(')',1)[1].split()[0] in ['Z','X']
    r=json.loads(path.read_text());assert r['rounds']==4 and r['discarded']==[1] and r['averaged']==[2,3,4]
    assert r['orders']==['AB','BA','AB','BA']
    assert r['GGUF_sha256']=='4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e'
    assert r['native_source_sha256']=='fb72d31e6d3f91e076fe9721744eacb7c379ffe0612c57e3c48ae4a25c20027c'
    assert r['model_packer_source_sha256']=='c1243e267d373490cf325d14d9431a80d163c313b56d6f217c83cd7f1a74f15e'
    assert r['portable_fusion_source_sha256']=='5daca6d9330b9b32a75dbcc8076a8c17dc433402509a87dff06a27ea1f22eeb5'
    assert r['default_pending_MG_policy_unchanged'] and not r['automatic_deployment'] and not r['same_arithmetic_claim']
    release=Path(r['public_source_release']);manifest=json.loads((release/'public-source-manifest.json').read_text())
    assert digest(release/'public-source-manifest.json')==r['public_source_manifest_sha256']
    for name,proof in manifest['files'].items():
        assert digest(release/name)==proof['sha256']
        if name.startswith('candidates_w4a16_v8/'):assert digest(source/name)==proof['sha256']
    def numeric(n):
        assert n['deterministic'] and math.isfinite(n['relative_rms']) and n['relative_rms']<.005
    def timing(t):
        assert t['orders']==[['reference','candidate'],['candidate','reference'],['reference','candidate'],['candidate','reference']]
        for label in ['reference','candidate']:
            values=t['samples_us'][label];assert len(values)==4 and all(math.isfinite(x) and x>0 for x in values)
            assert math.isclose(statistics.mean(values[1:]),t['means_us'][label],rel_tol=1e-9)
            assert math.isclose(statistics.stdev(values[1:]),t['sample_sd_us'][label],rel_tol=1e-9)
        assert math.isclose(t['candidate_over_reference'],t['means_us']['candidate']/t['means_us']['reference'],rel_tol=1e-9)
    assert len(r['cases'])==6 and {(c['role'],c['M']) for c in r['cases']}=={(role,m) for role in ['ffn_gate','ffn_up','ffn_gate_up'] for m in [8,16]}
    pairs=0
    for c in r['cases']:
        assert c['K']==6656 and c['N']==(39936 if c['role']=='ffn_gate_up' else 19968)
        numeric(c['reference_numerics']);assert len(c['packed_weight_sha256'])==5
        assert len(c['candidates'])==(2 if c['M']==8 else 3)
        for candidate in c['candidates']:
            assert candidate['config']['bucket']==c['M'];numeric(candidate['numerics']);timing(candidate['paired_timing']);pairs+=1
    assert pairs==15 and len(r['AOTI_cases'])==2 and {c['role'] for c in r['AOTI_cases']}=={'ffn_gate','ffn_gate_up'}
    routes={};decisions=[]
    for c in r['AOTI_cases']:
        assert c['bucket']==8 and c['dynamic_M_min']==5 and c['dynamic_M_max']==8 and c['pending_policy_not_promoted']
        assert c['AOTI_timing_policy']=='unchanged native cuda_graph_timing.time_with_cuda_graph_us; cold L2 per call; 4-32 calls in one graph; minimum GPU event interval'
        assert c['AOTI_invocation_policy']=='official AOTInductorModelContainerRunSingleThreaded with explicit current CUDA stream'
        assert {v['m'] for v in c['tail_numerics']}=={5,7,8} and len(c['tail_numerics'])==3
        for tail in c['tail_numerics']:
            assert set(tail['checks'])=={'reference','candidate'}
            for n in tail['checks'].values():numeric(n)
        assert set(c['packages'])=={'reference','candidate'}
        for package in c['packages'].values():
            f=Path(package['path']);assert f.resolve().is_relative_to((path.parent/'aoti-packages').resolve())
            assert f.stat().st_size==package['bytes'] and digest(f)==package['sha256']
        t=c['paired_timing'];timing(t)
        key=(8,c['n'],c['k']);assert key in [(8,19968,6656),(8,39936,6656)]
        use=t['candidate_over_reference']<.98
        if use:routes[key]=c['config_id']
        decisions.append({'shape':list(key),'config':c['config_id'],'candidate_over_reference':t['candidate_over_reference'],'use_candidate':use})
    assert routes,'No material current MG AOTI gain; keep native artifact'
    return routes,{'source_report_sha256':digest(path),'source_release_manifest_sha256':r['public_source_manifest_sha256'],
                   'real_eager_cases':6,'eager_pairs':15,'AOTI_pairs':2,'tail_checks':12,'route_decisions':decisions,
                   'whole_model_performance_memory_F1_acceptance_pending':True,'MG_M16_without_accepted_AOTI_route_remains_native':True}
