"""Recheck actual full-model protocols and tensors across a native SDK change."""
import argparse
import datetime
import json
from pathlib import Path
import statistics

import numpy as np
from independent_session_full_native_binding import sha


def review(before_job, after_job, profile):
    roots = [Path(before_job), Path(after_job)]
    docs = [json.loads((root/'result.json').read_text()) for root in roots]
    for key in ('model_PTE_sha256', 'weights_blob_sha256', 'model_geometry',
                'full_model_profile', 'initial_KV_capacity'):
        assert docs[0][key] == docs[1][key], key
    assert docs[0]['full_model_profile'] == profile
    assert docs[0]['initial_KV_capacity'] == 512
    assert docs[0]['probe_sha256'] != docs[1]['probe_sha256']
    assert all(doc['automatic_reclaim_requested'] and len(doc['rounds']) == 4 for doc in docs)
    count = 0
    tensors = []
    pids = set()
    evidence = {}
    timings = [[], []]
    for number in range(1, 5):
        pairs = []
        for arm, (root, doc) in enumerate(zip(roots, docs)):
            path = root/f'round-{number}/protocol.jsonl'
            assert sha(path) == doc['rounds'][number-1]['protocol_sha256']
            evidence[str(path.resolve())] = sha(path)
            operations = []
            pending = previous = None
            for line in path.open():
                row = json.loads(line)
                if 'request' in row:
                    assert pending is None
                    pending = row['request']
                    continue
                response = row['response']
                assert not response.get('failed')
                if response.get('ready'):
                    assert pending is None and previous is None
                    previous = response['snapshot']
                    continue
                assert pending and pending['op'] == response['op']
                counts = lambda snapshot: {m['method']: m['calls'] for m in snapshot['method_calls']}
                old, new = counts(previous), counts(response['snapshot'])
                delta = {k: new[k]-old[k] for k in new if new[k] != old[k]}
                if pending['op'] == 'step':
                    assert len(delta) == 1 and list(delta.values()) == [1]
                    assert len(response['argmax']) == len(pending['select'])
                else:
                    assert not delta
                assert pending['op'] != 'compact'
                operations.append((pending, response))
                pending, previous = None, response['snapshot']
            assert pending is None
            actual_pids = {response['snapshot']['pid'] for _, response in operations}
            assert len(actual_pids) == 1 and not pids.intersection(actual_pids)
            pids.update(actual_pids)
            assert [s['after']['physical_columns'] for s in doc['rounds'][number-1]['stages']] == [2,3,5,9,17,2]
            assert doc['rounds'][number-1]['stages'][-1]['after']['live_sequences'] == 1
            assert len(doc['rounds'][number-1]['post_shrink_method_checks']) == 35
            for check in doc['rounds'][number-1]['post_shrink_method_checks']:
                assert check['method_delta'] == {f"forward_{check['static_width']}": 1}
                assert check['snapshot']['physical_columns'] == 2
            timings[arm].append(sum(response['elapsed_seconds'] for request,response in operations if request['op']=='step'))
            pairs.append(operations)
        assert len(pairs[0]) == len(pairs[1])
        round_steps = 0
        for (left_req, left), (right_req, right) in zip(*pairs):
            assert {k:v for k,v in left_req.items() if k!='tag'} == {k:v for k,v in right_req.items() if k!='tag'}
            if left_req['op'] != 'step':
                continue
            round_steps += 1
            assert left['argmax'] == right['argmax']
            assert len(left['logits']) == len(right['logits'])
            for left_part, right_part in zip(left['logits'], right['logits']):
                assert left_part['shape'] == right_part['shape']
                assert left_part['selected_indices'] == right_part['selected_indices']
                arrays, hashes = [], []
                for root, part in zip(roots, (left_part, right_part)):
                    path = Path(part['file'])
                    assert path.resolve().is_relative_to(root.resolve())
                    assert path.stat().st_size == int(np.prod(part['shape'])) * 4
                    arrays.append(np.fromfile(path, dtype='<f4'))
                    hashes.append(sha(path))
                    evidence[str(path.resolve())] = sha(path)
                assert hashes[0] == hashes[1] and np.isfinite(arrays[0]).all()
                assert np.array_equal(*arrays)
                tensors.append(dict(round=number, shape=left_part['shape'], SHA256=hashes[0]))
        assert round_steps == (447 if profile=='muse-glimmer-52layers' else 489)
        count += round_steps
    assert len(pids) == 8 and len(tensors) == (39 if profile=='muse-glimmer-52layers' else 42)
    arms = [dict(mean=statistics.mean(values[1:]), sample_SD=statistics.stdev(values[1:]),
                 rounds2_4=values[1:]) for values in timings]
    bound = 3*((arms[0]['sample_SD']**2+arms[1]['sample_SD']**2)/3)**0.5
    delta = arms[1]['mean']-arms[0]['mean']
    for root in roots:
        evidence[str((root/'result.json').resolve())] = sha(root/'result.json')
    return dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        full_model_profile=profile, same_PTE_and_PTD_verified=True,
        all_actual_inputs_and_selected_outputs_equal=True, forward_steps_per_arm=count,
        full_FP32_tensors_bitwise_equal=len(tensors), fresh_native_PIDs=sorted(pids),
        automatic_reclaim_and_padding_verified=True, evidence_files_SHA256=evidence,
        native_step_seconds=dict(before=arms[0], after=arms[1], delta_seconds=delta,
            combined_3SE_seconds=bound, positive_signal=delta>bound),
        artifact_SHA256={key:docs[0][key] for key in ('model_PTE_sha256','weights_blob_sha256')},
        native_gate_SHA256=[d['native_gate_sha256'] for d in docs],
        probe_SHA256=[d['probe_sha256'] for d in docs],
        full_task_F1_reviewed=False, full_model_qualified=False, main_candidate_accepted=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('before-job','after-job','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--profile',choices=['muse-glimmer-52layers','gemma4-48layers'],required=True)
    args=parser.parse_args();assert not args.output.exists()
    result=review(args.before_job,args.after_job,args.profile)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('forward_steps_per_arm','full_FP32_tensors_bitwise_equal')}))


if __name__=='__main__':main()
