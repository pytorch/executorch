"""Reuse verified full PTEs with the separately qualified empty-owner SDK fix.

Keep the original export gates, models and SDK unchanged. This creates aliases
and provenance for full-model qualification; it does not accept main results.
"""
import argparse
import datetime
import json
import os
from pathlib import Path

from independent_session_full_native_binding import (
    alias_runtime, sha, verify_qualification, verify_runtime)
from independent_session_mini_profiles import FULL_PROFILES, validate_full_audit


POLICIES = {
    'paired-KV-storage': ('independent_session_paired_KV_storage', 'paired_KV_storage_spec_SHA256'),
    'isolated-KV-growth-slack': ('independent_session_isolated_KV_growth_slack', 'isolated_KV_growth_slack_spec_SHA256'),
    'isolated-KV-pool': ('independent_session_isolated_KV_pool', 'isolated_KV_pool_spec_SHA256'),
    'empty-owner-reuse': ('independent_session_empty_replacement', 'empty_owner_reuse_spec_SHA256'),
    'coalesced-growth-trim': ('independent_session_coalesced_growth_trim', 'coalesced_growth_trim_spec_SHA256'),
    'workspace-preserving-growth-trim': ('independent_session_workspace_preserving_growth_trim', 'workspace_preserving_growth_trim_spec_SHA256'),
}


def records(gate):
    change = gate['actual_native_source_change']
    return change['patches'] if 'patches' in change else [change]


def verify_transition(before_job, after_job, package, variant, transition='empty-owner-reuse'):
    before = verify_runtime(before_job, package)
    after = verify_runtime(after_job, package)
    assert transition in POLICIES
    policy, key = POLICIES[transition]
    fix_path = package/(policy + '_source.json')
    fix = json.loads(fix_path.read_text())
    assert sha(package/(policy + '.patch')) == fix['patch_SHA256']
    assert after['preserved_preceding_native_gate_SHA256'] == sha(before_job/'native-gate.json')
    assert after[key] == sha(fix_path)
    assert records(after) == records(before) + fix['variants'][variant]
    if transition != 'empty-owner-reuse':
        assert before.get('empty_owner_reuse_spec_SHA256')
        assert after['empty_owner_reuse_spec_SHA256'] == before['empty_owner_reuse_spec_SHA256']
        if transition == 'paired-KV-storage':
            assert before.get('isolated_KV_growth_slack_spec_SHA256')
            assert after['isolated_KV_growth_slack_spec_SHA256'] == before['isolated_KV_growth_slack_spec_SHA256']
            assert all(r['paired_private_KV_allocation_only'] and
                       r['weights_math_PTE_ABI_owner_capacity_sliding_copy_and_graph_invalidation_unchanged']
                       for r in fix['variants'][variant])
        elif transition == 'isolated-KV-growth-slack':
            assert before.get('isolated_KV_pool_spec_SHA256')
            assert after['isolated_KV_pool_spec_SHA256'] == before['isolated_KV_pool_spec_SHA256']
            assert all(r['private_KV_allocator_and_instrumentation_only'] and
                       r['weights_math_PTE_ABI_owner_capacity_sliding_copy_and_graph_invalidation_unchanged']
                       and r['requested_growth_slack_bytes']==64*2**20 and r['contraction_requested_slack_bytes']==0
                       for r in fix['variants'][variant])
        elif transition == 'isolated-KV-pool':
            assert all(r['private_KV_allocator_and_instrumentation_only'] and
                       r['weights_math_PTE_ABI_owner_capacity_sliding_copy_and_graph_invalidation_unchanged']
                       for r in fix['variants'][variant])
        else:
            assert all(r['allocator_policy_only'] and r['weights_math_ABI_sliding_and_graph_invalidation_unchanged']
                       for r in fix['variants'][variant])
    assert after['source_spec_SHA256'] == before['source_spec_SHA256']
    assert after['unchanged_controls'] == before['unchanged_controls']
    assert after['independent_session_tests'] == before['independent_session_tests']
    assert Path(before['source']).resolve() != Path(after['source']).resolve()
    return before, after


def verify_transition_chain(before_job, after_job, package, variant, transition, via_job=None, intermediate_jobs=None):
    """Verify each actual SDK hop without qualifying an intermediate runtime."""
    if intermediate_jobs:
        assert transition == 'paired-KV-storage' and via_job is None
        assert len(intermediate_jobs) == 3, 'Require actual workspace, private-pool and bounded-slack SDKs'
        paths = [Path(job).resolve() for job in (before_job,*intermediate_jobs,after_job)]
        assert len(set(paths)) == 5
        first = previous = final = None
        for i,policy in enumerate(('workspace-preserving-growth-trim','isolated-KV-pool',
                                   'isolated-KV-growth-slack','paired-KV-storage')):
            before,after=verify_transition(paths[i],paths[i+1],Path(package),variant,policy)
            if previous is not None:assert previous==before
            if first is None:first=before
            previous=final=after
        return first,final
    assert transition != 'paired-KV-storage', 'Paired full rebind requires all three actual intermediate SDKs'
    if via_job is None:
        return verify_transition(before_job, after_job, package, variant, transition)
    assert transition == 'isolated-KV-pool', 'Only the explicit private-pool source chain uses an intermediate SDK'
    paths = [Path(job).resolve() for job in (before_job, via_job, after_job)]
    assert len(set(paths)) == 3, 'Each native source hop must be distinct'
    before, middle = verify_transition(Path(before_job), Path(via_job), Path(package), variant,
                                       'workspace-preserving-growth-trim')
    actual_middle, after = verify_transition(Path(via_job), Path(after_job), Path(package), variant, transition)
    assert middle == actual_middle
    return before, after


def verify_export(artifact, export_job, before, profile, arm):
    artifact, export_job = Path(artifact), Path(export_job)
    audit_path = artifact/'independent-full-artifact-audit.json'
    audit = json.loads(audit_path.read_text())
    validate_full_audit(audit, profile, arm == 'independent')
    assert audit['native_gate_sha256'] == sha(export_job/'native-gate.json')
    export_gate = json.loads((export_job/'native-gate.json').read_text())
    for key in ('source', 'source_spec_SHA256', 'native_tests_xml_SHA256',
                'unchanged_controls', 'actual_native_source_change'):
        assert export_gate[key] == before[key], key
    assert export_gate['artifact_export_and_native_SDK_sources_are_separately_verified']
    for path, expected in export_gate['binaries'].items():
        assert sha(path) == expected, path
    assert audit['bulk_prefill_A16_original_native_decode']
    if profile == 'gemma4-48layers':
        assert audit['safe_ring_rows'] == 1535
        assert audit['precision_kernel_sha256'] == '10affaf3cd36f17a68f8336e764e24b16e54dd509d89577c9009445b2f41cf9b'
    for name, expected in audit['artifacts'].items():
        path = artifact/name
        assert path.resolve().is_relative_to(artifact.resolve())
        assert sha(path) == expected['sha256'], name
    return audit


def verify_rebind(proof_path, runtime_job, package):
    """Recheck actual old/new source chains and immutable artifacts at use time."""
    proof = json.loads(Path(proof_path).read_text())
    transition = proof.get('native_transition', 'empty-owner-reuse')
    assert proof['only_empty_owner_native_replacement'] == (transition == 'empty-owner-reuse')
    assert not proof['full_model_qualified'] and not proof['main_candidate_accepted']
    assert Path(runtime_job).resolve() == Path(proof['runtime_job']).resolve()
    for name, expected in proof['evidence_files_SHA256'].items():
        assert sha(name) == expected, name
    before, after = verify_transition_chain(Path(proof['export_runtime_job']),
        Path(runtime_job), Path(package), proof['source_variant'], transition,
        proof.get('transition_via_native_job'),proof.get('transition_intermediate_native_jobs'))
    if proof.get('transition_intermediate_native_jobs'):
        assert len(proof['transition_intermediate_native_gate_SHA256'])==3
        for job,expected in zip(proof['transition_intermediate_native_jobs'],proof['transition_intermediate_native_gate_SHA256']):
            assert sha(Path(job)/'native-gate.json')==expected
    if proof.get('transition_via_native_job'):
        assert sha(Path(proof['transition_via_native_job'])/'native-gate.json') == proof['transition_via_native_gate_SHA256']
    export_runtime = before
    original_export_runtime_job = proof['export_runtime_job']
    preceding = None
    if proof.get('preceding_full_rebind_proof'):
        assert transition in POLICIES and transition != 'empty-owner-reuse'
        preceding_path = Path(proof['preceding_full_rebind_proof'])
        assert preceding_path.resolve() != Path(proof_path).resolve()
        assert sha(preceding_path) == proof['preceding_full_rebind_proof_SHA256']
        preceding = verify_rebind(preceding_path, Path(proof['export_runtime_job']), package)
        assert preceding['full_model_profile'] == proof['full_model_profile']
        assert preceding['artifacts'] == proof['artifacts']
        assert preceding.get('native_transition', 'empty-owner-reuse') == 'empty-owner-reuse'
        export_runtime = verify_runtime(Path(preceding['export_runtime_job']), package)
        original_export_runtime_job = preceding.get('original_model_export_runtime_job', preceding['export_runtime_job'])
    if 'original_model_export_runtime_job' in proof:
        assert Path(proof['original_model_export_runtime_job']).resolve() == Path(original_export_runtime_job).resolve()
    profile = proof['full_model_profile']
    mini_profile = 'gemma4-first6' if profile == 'gemma4-48layers' else 'muse-glimmer-first4'
    verify_qualification(proof['mini_qualification'], mini_profile,
        FULL_PROFILES[profile]['GGUF_SHA256'], runtime_job)
    for arm, item in proof['artifacts'].items():
        assert arm in ('global', 'independent')
        verify_export(item['artifact'], item['export_job'], export_runtime, profile, arm)
    assert proof['native_runtime_gate_SHA256'] == sha(Path(runtime_job)/'native-gate.json')
    return proof


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('package', 'export-runtime-job', 'runtime-job',
                 'mini-qualification', 'independent-artifact',
                 'independent-export-job', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--global-artifact', type=Path)
    parser.add_argument('--global-export-job', type=Path)
    parser.add_argument('--preceding-full-rebind-proof', type=Path,
                        help='Preserved empty-owner rebind linking these exact models to the preceding runtime')
    parser.add_argument('--transition-via-native-job', type=Path,
                        help='Actual workspace-preserving SDK between the export runtime and private KV SDK')
    parser.add_argument('--transition-intermediate-native-jobs',type=Path,nargs=3,
                        help='Actual workspace/private-pool/bounded-slack SDKs preceding paired K/V storage')
    parser.add_argument('--native-transition', choices=POLICIES,
                        default='empty-owner-reuse')
    parser.add_argument('--full-model-profile', choices=FULL_PROFILES, required=True)
    args = parser.parse_args()
    assert bool(args.global_artifact) == bool(args.global_export_job)
    package = args.package.resolve()
    profile = args.full_model_profile
    gemma = profile == 'gemma4-48layers'
    variant = ('Gemma-stable-metadata' if gemma else 'MG-flat-copy') if args.native_transition == 'empty-owner-reuse' else (
        'Gemma-empty-owner-reuse' if gemma else 'MG-empty-owner-reuse')
    before, after = verify_transition_chain(args.export_runtime_job, args.runtime_job, package, variant,
                                           args.native_transition, args.transition_via_native_job,
                                           args.transition_intermediate_native_jobs)
    qualification = verify_qualification(args.mini_qualification,
        'gemma4-first6' if gemma else 'muse-glimmer-first4',
        FULL_PROFILES[profile]['GGUF_SHA256'], args.runtime_job)
    artifacts = {'independent': dict(artifact=str(args.independent_artifact.resolve()),
        export_job=str(args.independent_export_job.resolve()))}
    if args.global_artifact:
        artifacts['global'] = dict(artifact=str(args.global_artifact.resolve()),
            export_job=str(args.global_export_job.resolve()))
    export_runtime = before
    original_export_runtime_job = str(args.export_runtime_job.resolve())
    preceding = None
    if args.preceding_full_rebind_proof:
        assert args.native_transition in POLICIES and args.native_transition != 'empty-owner-reuse'
        preceding = verify_rebind(args.preceding_full_rebind_proof, args.export_runtime_job, package)
        assert preceding['full_model_profile'] == profile and preceding['artifacts'] == artifacts
        assert preceding.get('native_transition', 'empty-owner-reuse') == 'empty-owner-reuse'
        export_runtime = verify_runtime(Path(preceding['export_runtime_job']), package)
        original_export_runtime_job = preceding.get('original_model_export_runtime_job', preceding['export_runtime_job'])
    audits = {arm: verify_export(item['artifact'], item['export_job'], export_runtime, profile, arm)
              for arm, item in artifacts.items()}
    assert not args.output.exists(), 'Preserve preceding outputs'
    args.output.mkdir(parents=True)
    evidence = {str(args.mini_qualification.resolve()): sha(args.mini_qualification),
        str(args.export_runtime_job.resolve()/'native-gate.json'): sha(args.export_runtime_job/'native-gate.json'),
        str(args.runtime_job.resolve()/'native-gate.json'): sha(args.runtime_job/'native-gate.json'),
        str(package/(POLICIES[args.native_transition][0] + '_source.json')):
        sha(package/(POLICIES[args.native_transition][0] + '_source.json'))}
    if args.transition_via_native_job:
        for path in (args.transition_via_native_job.resolve()/'native-gate.json',
                     package/'independent_session_workspace_preserving_growth_trim_source.json'):
            evidence[str(path)] = sha(path)
    if args.transition_intermediate_native_jobs:
        for job in args.transition_intermediate_native_jobs:
            path=job.resolve()/'native-gate.json';evidence[str(path)]=sha(path)
        for policy in ('workspace-preserving-growth-trim','isolated-KV-pool','isolated-KV-growth-slack'):
            path=package/(POLICIES[policy][0]+'_source.json');evidence[str(path)]=sha(path)
    for item in artifacts.values():
        for path in (Path(item['artifact'])/'independent-full-artifact-audit.json',
                     Path(item['export_job'])/'native-gate.json'):
            evidence[str(path)] = sha(path)
    if args.preceding_full_rebind_proof:
        evidence[str(args.preceding_full_rebind_proof.resolve())] = sha(args.preceding_full_rebind_proof)
    proof = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        full_model_profile=profile, source_variant=variant,
        export_runtime_job=str(args.export_runtime_job.resolve()),
        original_model_export_runtime_job=original_export_runtime_job,
        runtime_job=str(args.runtime_job.resolve()),
        mini_qualification=str(args.mini_qualification.resolve()),
        native_runtime_gate_SHA256=sha(args.runtime_job/'native-gate.json'),
        artifacts=artifacts, evidence_files_SHA256=evidence,
        native_transition=args.native_transition,
        only_empty_owner_native_replacement=args.native_transition == 'empty-owner-reuse',
        original_models_and_export_SDK_preserved=True,
        no_GEMM_attention_or_model_method_change=True,
        full_model_qualified=False, main_candidate_accepted=False)
    if args.transition_via_native_job:
        proof.update(transition_via_native_job=str(args.transition_via_native_job.resolve()),
                     transition_via_native_gate_SHA256=sha(args.transition_via_native_job/'native-gate.json'))
    if args.transition_intermediate_native_jobs:
        proof.update(transition_intermediate_native_jobs=[str(job.resolve()) for job in args.transition_intermediate_native_jobs],
                     transition_intermediate_native_gate_SHA256=[sha(job/'native-gate.json') for job in args.transition_intermediate_native_jobs])
    if args.preceding_full_rebind_proof:
        proof.update(preceding_full_rebind_proof=str(args.preceding_full_rebind_proof.resolve()),
                     preceding_full_rebind_proof_SHA256=sha(args.preceding_full_rebind_proof))
    proof_path = args.output/'full-runtime-rebind.json'
    proof_path.write_text(json.dumps(proof, indent=2)+'\n')
    verify_rebind(proof_path, args.runtime_job, package)
    alias_runtime(args.runtime_job, package, args.output, export_runtime['source'], extra={
        'full_runtime_rebind_proof': str(proof_path.resolve()),
        'full_runtime_rebind_proof_SHA256': sha(proof_path)})
    for arm, item in artifacts.items():
        (args.output/('full-'+arm+'-artifact')).symlink_to(item['artifact'])
    state = dict(status='full-artifacts-ready-for-qualification',
        controller_pid=os.getpid(), mini_gate_passed=True,
        all_full_artifact_hashes_verified=True, full_profile=profile,
        full_audit_SHA256={arm: sha(Path(item['artifact'])/'independent-full-artifact-audit.json')
                           for arm, item in artifacts.items()},
        full_runtime_rebind_proof_SHA256=sha(proof_path),
        current_native_and_original_export_provenance_separate=True,
        full_model_qualified=False, main_candidate_accepted=False)
    (args.output/'progress.json').write_text(json.dumps(state, indent=2)+'\n')
    print(json.dumps(state))


if __name__ == '__main__':
    main()
