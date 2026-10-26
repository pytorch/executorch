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


def records(gate):
    change = gate['actual_native_source_change']
    return change['patches'] if 'patches' in change else [change]


def verify_transition(before_job, after_job, package, variant):
    before = verify_runtime(before_job, package)
    after = verify_runtime(after_job, package)
    fix_path = package/'independent_session_empty_replacement_source.json'
    fix = json.loads(fix_path.read_text())
    assert sha(package/'independent_session_empty_replacement.patch') == fix['patch_SHA256']
    assert after['preserved_preceding_native_gate_SHA256'] == sha(before_job/'native-gate.json')
    assert after['empty_owner_reuse_spec_SHA256'] == sha(fix_path)
    assert records(after) == records(before) + fix['variants'][variant]
    assert after['source_spec_SHA256'] == before['source_spec_SHA256']
    assert after['unchanged_controls'] == before['unchanged_controls']
    assert after['independent_session_tests'] == before['independent_session_tests']
    assert Path(before['source']).resolve() != Path(after['source']).resolve()
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
    assert proof['only_empty_owner_native_replacement']
    assert not proof['full_model_qualified'] and not proof['main_candidate_accepted']
    assert Path(runtime_job).resolve() == Path(proof['runtime_job']).resolve()
    for name, expected in proof['evidence_files_SHA256'].items():
        assert sha(name) == expected, name
    before, after = verify_transition(Path(proof['export_runtime_job']),
        Path(runtime_job), Path(package), proof['source_variant'])
    profile = proof['full_model_profile']
    mini_profile = 'gemma4-first6' if profile == 'gemma4-48layers' else 'muse-glimmer-first4'
    verify_qualification(proof['mini_qualification'], mini_profile,
        FULL_PROFILES[profile]['GGUF_SHA256'], runtime_job)
    for arm, item in proof['artifacts'].items():
        assert arm in ('global', 'independent')
        verify_export(item['artifact'], item['export_job'], before, profile, arm)
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
    parser.add_argument('--full-model-profile', choices=FULL_PROFILES, required=True)
    args = parser.parse_args()
    assert bool(args.global_artifact) == bool(args.global_export_job)
    package = args.package.resolve()
    profile = args.full_model_profile
    gemma = profile == 'gemma4-48layers'
    variant = 'Gemma-stable-metadata' if gemma else 'MG-flat-copy'
    before, after = verify_transition(args.export_runtime_job, args.runtime_job, package, variant)
    qualification = verify_qualification(args.mini_qualification,
        'gemma4-first6' if gemma else 'muse-glimmer-first4',
        FULL_PROFILES[profile]['GGUF_SHA256'], args.runtime_job)
    artifacts = {'independent': dict(artifact=str(args.independent_artifact.resolve()),
        export_job=str(args.independent_export_job.resolve()))}
    if args.global_artifact:
        artifacts['global'] = dict(artifact=str(args.global_artifact.resolve()),
            export_job=str(args.global_export_job.resolve()))
    audits = {arm: verify_export(item['artifact'], item['export_job'], before, profile, arm)
              for arm, item in artifacts.items()}
    assert not args.output.exists(), 'Preserve preceding outputs'
    args.output.mkdir(parents=True)
    evidence = {str(args.mini_qualification.resolve()): sha(args.mini_qualification),
        str(args.export_runtime_job.resolve()/'native-gate.json'): sha(args.export_runtime_job/'native-gate.json'),
        str(args.runtime_job.resolve()/'native-gate.json'): sha(args.runtime_job/'native-gate.json'),
        str(package/'independent_session_empty_replacement_source.json'): sha(package/'independent_session_empty_replacement_source.json')}
    for item in artifacts.values():
        for path in (Path(item['artifact'])/'independent-full-artifact-audit.json',
                     Path(item['export_job'])/'native-gate.json'):
            evidence[str(path)] = sha(path)
    proof = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        full_model_profile=profile, source_variant=variant,
        export_runtime_job=str(args.export_runtime_job.resolve()),
        runtime_job=str(args.runtime_job.resolve()),
        mini_qualification=str(args.mini_qualification.resolve()),
        native_runtime_gate_SHA256=sha(args.runtime_job/'native-gate.json'),
        artifacts=artifacts, evidence_files_SHA256=evidence,
        only_empty_owner_native_replacement=True,
        original_models_and_export_SDK_preserved=True,
        no_GEMM_attention_or_model_method_change=True,
        full_model_qualified=False, main_candidate_accepted=False)
    proof_path = args.output/'full-runtime-rebind.json'
    proof_path.write_text(json.dumps(proof, indent=2)+'\n')
    verify_rebind(proof_path, args.runtime_job, package)
    alias_runtime(args.runtime_job, package, args.output, before['source'], extra={
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
