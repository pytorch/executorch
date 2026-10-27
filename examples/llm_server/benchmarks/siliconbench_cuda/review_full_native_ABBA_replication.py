"""Recompute two real full-model ABBA studies with six retained samples each.

Every study still has four rounds and discards its first round. Keep the
original signal; use the unchanged three-standard-error rule on all retained
samples. This review cannot qualify a model without the other full gates.
"""
import argparse,datetime,hashlib,json,math,statistics,subprocess,sys
from pathlib import Path
from independent_session_full_native_binding import sha

METRICS=('cold_first_four_seconds','mean_stable_native_ms')


def cells_SHA256(cells):
    return hashlib.sha256(json.dumps(cells,sort_keys=True,allow_nan=False).encode()).hexdigest()


def verify_user_performance_acceptance(path,report):
    """Accept only explicitly approved, unchanged performance evidence.

    This does not change a statistical result or any correctness requirement.
    The decision file belongs to the experiment, not the public checkout.
    """
    decision=json.loads(Path(path).read_text())
    assert decision['schema']==1 and decision['explicit_user_acceptance'] is True
    assert decision['decision']=='accept_current_Gemma_full_native_performance_for_main_collection'
    assert decision['full_profile']=='gemma4-48layers'
    assert decision['correctness_exceptions_allowed'] is False
    assert decision['user_instruction'] and decision['UTC']
    assert decision['GPU_UUID']==report['qualification_GPU_UUID']
    assert decision['accepted_cells_SHA256']==cells_SHA256(report['cells'])
    assert decision['accepted_signals']==report['positive_performance_signals']
    assert decision['accepted_signals']==[dict(arm='independent',case='decode8',metric='mean_stable_native_ms')]
    assert decision['study_result_SHA256']==report['study_result_SHA256']
    for item in report['study_result_SHA256']:assert sha(item['path'])==item['SHA256']
    reviews=[json.loads(Path(p).read_text()) for p in report['recomputed_reviews']]
    for reviewed in reviews:
        assert decision['native_gate_SHA256']==reviewed['native_gate_SHA256'][1]
        assert decision['artifacts']==reviewed['artifacts']
        assert reviewed['native_change_preserves_observed_model_numerics']
        assert reviewed['full_saved_FP32_tensors_equal']==9
        assert reviewed['selected_argmax_values_equal']==3672
    return decision


def pool(reviews):
    assert len(reviews)==2
    cells=[]
    tables=[{(c['arm'],c['case']):c for c in r['cells']} for r in reviews]
    assert set(tables[0])==set(tables[1]) and len(tables[0])==9
    for key in sorted(tables[0]):
        cell=dict(arm=key[0],case=key[1]);assert key[0]=='independent'
        for metric in METRICS:
            stats={}
            for side in ('before','after'):
                values=[v for t in tables for v in t[key][metric][side]['values']]
                assert len(values)==6 and all(math.isfinite(v) and v>0 for v in values)
                stats[side]=dict(values=values,mean=statistics.mean(values),sample_SD=statistics.stdev(values))
            delta=stats['after']['mean']-stats['before']['mean']
            threshold=3*math.sqrt(sum(stats[s]['sample_SD']**2/6 for s in stats))
            cell[metric]=dict(**stats,delta=delta,delta_percent=100*delta/stats['before']['mean'],combined_3SE=threshold,positive_signal=delta>threshold)
        cells.append(cell)
    return cells


def verify_report(path,package,UUID):
    d=json.loads(Path(path).read_text())
    assert d['producer_source_SHA256']==sha(Path(package)/Path(__file__).name)
    assert d['qualification_GPU_UUID']==UUID and d['retained_values_each_native']==6
    for name,expected in d['evidence_files_SHA256'].items():assert sha(name)==expected,name
    reviews=[json.loads(Path(name).read_text()) for name in d['recomputed_reviews']]
    assert pool(reviews)==d['cells']
    signals=[dict(arm=c['arm'],case=c['case'],metric=m) for c in d['cells'] for m in METRICS if c[m]['positive_signal']]
    assert signals==d['positive_performance_signals']
    if signals:
        assert d.get('performance_accepted_by_user') and not d['strict_statistical_performance_pass']
        assert sha(d['performance_acceptance'])==d['performance_acceptance_SHA256']
        verify_user_performance_acceptance(d['performance_acceptance'],d)
    return d


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('original-job','confirmation-job','package','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--performance-acceptance',type=Path)
    a=p.parse_args();assert a.original_job.resolve()!=a.confirmation_job.resolve() and not a.output.exists();a.output.mkdir(parents=True)
    from write_MG_full_independent_KV_qualification import current_GPU_UUID
    UUID=current_GPU_UUID();documents=[];reviews=[];evidence={};paths=[]
    for index,root in enumerate((a.original_job,a.confirmation_job)):
        assert json.loads((root/'progress.json').read_text())['status']=='interleaved-native-control-collected'
        d=json.loads((root/'result.json').read_text());documents.append(d)
        assert d['full_model_profile']=='gemma4-48layers' and d['collected_arms']==['independent']
        assert d['four_rounds_discard_first'] and d['ABBA_native_order'] and d['initial_KV_capacity']==512
        assert d['unchanged_PTE_PTD_and_tokens'] and len(d['actual_child_results'])==8
        assert sha(root/'GPU-clocks-power.jsonl')==d['clock_power_samples_SHA256']
        assert {json.loads(line)['actual_GPU_query'].split(',')[0].strip() for line in (root/'GPU-clocks-power.jsonl').read_text().splitlines()}=={UUID}
        for c in d['actual_child_results']:assert sha(c['result'])==c['SHA256']
        output=a.output/('study-'+str(index)+'-raw-review.json')
        subprocess.run([sys.executable,'-B',str(a.package/'review_independent_session_same_artifact_native_change.py'),'--before-job',str(root/'before'),'--after-job',str(root/'after'),'--output',str(output)],check=True,timeout=600)
        review=json.loads(output.read_text());observed=json.loads((root/'same-PTE-native-independent-review.json').read_text())
        assert review['cells']==observed['cells'] and review['records']==observed['records']
        assert review['full_model_profile']=='gemma4-48layers' and review['full_saved_FP32_tensors_equal']==9 and review['selected_argmax_values_equal']==3672
        reviews.append(review);paths.append(str(output.resolve()))
        for f in root.rglob('*'):
            if f.is_file() and not f.is_symlink() and f.suffix in ('.json','.jsonl','.f32','.log','.sqlite','.nsys-rep'):evidence[str(f.resolve())]=sha(f)
        evidence[str(output.resolve())]=sha(output)
    for key in ('artifacts','native_gate_SHA256','probe_SHA256'):
        assert reviews[0][key]==reviews[1][key],key
    # Compare actual arguments while retaining each study's separate output.
    def signature(d):
        result=[]
        for row in sorted(d['actual_child_results'],key=lambda r:(r['round'],r['native'])):
            argv=list(row['command']);position=argv.index('--output');argv[position+1]='separate-study-output'
            result.append((row['round'],row['native'],argv))
        return result
    assert signature(documents[0])==signature(documents[1])
    assert not set(r['result'] for r in documents[0]['actual_child_results']) & set(r['result'] for r in documents[1]['actual_child_results'])
    cells=pool(reviews);signals=[dict(arm=c['arm'],case=c['case'],metric=m) for c in cells for m in METRICS if c[m]['positive_signal']]
    report=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),producer_source_SHA256=sha(Path(__file__)),qualification_GPU_UUID=UUID,recomputed_reviews=paths,cells=cells,positive_performance_signals=signals,original_matched_signals=documents[0]['positive_performance_signals'],confirmation_matched_signals=documents[1]['positive_performance_signals'],retained_values_each_native=6,each_study_four_rounds_discard_first=True,original_gate_preserved=True,statistical_rule_changed=False,no_capture_or_stable_waiver=True,evidence_files_SHA256=evidence)
    report['study_result_SHA256']=[dict(path=str((root/'result.json').resolve()),SHA256=sha(root/'result.json')) for root in (a.original_job,a.confirmation_job)]
    report['strict_statistical_performance_pass']=not signals
    report['performance_accepted_by_user']=False
    if a.performance_acceptance:
        assert signals,'Explicit exception must refer to the measured signal'
        verify_user_performance_acceptance(a.performance_acceptance,report)
        report.update(performance_accepted_by_user=True,performance_acceptance=str(a.performance_acceptance.resolve()),performance_acceptance_SHA256=sha(a.performance_acceptance))
        evidence[str(a.performance_acceptance.resolve())]=sha(a.performance_acceptance)
    (a.output/'replication-review.json').write_text(json.dumps(report,indent=2)+'\n')
    assert not signals or report['performance_accepted_by_user'],'Actual pooled full native timing still has signals; preserve raw and do not qualify'
    verify_report(a.output/'replication-review.json',a.package,UUID)
    print(json.dumps(dict(actual_two_study_full_ABBA_pass=not signals,performance_accepted_by_user=report['performance_accepted_by_user'],original_gate_preserved=True)))


if __name__=='__main__':main()
