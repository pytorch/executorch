"""Verify full Gemma artifacts and the already qualified HTTP/native adapter.

This binding does not promote a candidate or replace task correctness checks.
It preserves export-source, runtime-source and sampler provenance separately.
"""
import json
from pathlib import Path
from independent_session_full_native_binding import verify_runtime, sha
from independent_session_mini_profiles import validate_full_audit


def verified_gemma_full_http(args, package):
    assert args.full_model_profile == 'gemma4-48layers'
    assert args.mini_profile == 'gemma4-first6'
    assert args.full_native_job and args.memory_worker_proof
    root = args.preparation_job
    prep = json.loads((root/'progress.json').read_text())
    assert prep['status'] == 'full-artifacts-ready-for-qualification'
    assert prep['mini_gate_passed'] and prep['all_full_artifact_hashes_verified']
    actual = verify_runtime(args.full_native_job, package)
    gate_path = root/'native-gate.json'
    gate = json.loads(gate_path.read_text())
    for key in ('source', 'source_spec_SHA256', 'native_tests_xml_SHA256',
                'unchanged_controls', 'actual_native_source_change'):
        assert gate[key] == actual[key], key
    for name, expected in gate['binaries'].items():
        assert sha(name) == expected, name
    assert gate['artifact_export_and_native_SDK_sources_are_separately_verified']
    native_source = Path(actual['source'])
    proof = json.loads(args.memory_worker_proof.read_text())
    assert proof['native_gate_SHA256'] == sha(args.full_native_job/'native-gate.json')
    assert proof['source'] == str(native_source)
    for key in ('existing_Gemma_llm_server_adapter_preserved',
                'scheduler_stops_tokenizer_transport_unchanged',
                'original_source_and_SDK_unchanged', 'no_inference_kernel_change',
                'sampler_does_not_synchronize_GPU', 'optional_native_sampler_defaults_off'):
        assert proof[key], key
    assert proof['original_Gemma_worker_source_SHA256'] == sha(package/'gemma_runtime_columns_worker.cpp')
    assert proof['optional_sampler_header_SHA256'] == sha(package/'kv_memory_sampler.h')
    worker = Path(proof['instrumented_worker'])
    assert sha(worker) == proof['instrumented_worker_SHA256']
    for name, expected in proof['unchanged_SDK_libraries'].items():
        assert sha(name) == expected, name
    header = native_source/'backends/cuda/batching/cuda_executor.h'
    assert sha(header) == proof['qualified_executor_options_header_SHA256']
    legacy = not any(name in header.read_text() for name in
                     ('idle_kv_compaction_ms', 'idle_kv_spare_columns'))
    assert proof['legacy_idle_option_assignments_omitted'] == legacy
    receipt_path = args.Gemma_source_preparation_job/'source-preparation.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['original_source_and_binaries_unchanged']
    source = Path(receipt['new_source'])
    for name, expected in receipt['source_files'].items():
        assert sha(source/name) == expected, name
    from prepare_gemma_canonical_tokenizer import FILES
    for name, (expected, count) in FILES.items():
        f = args.tokenizer/name
        assert f.stat().st_size == count and sha(f) == expected, name
    assert args.http_python_dependencies_job
    dependency_job = args.http_python_dependencies_job
    assert json.loads((dependency_job/'progress.json').read_text())['status'] == 'Gemma-HTTP-Python-dependencies-ready'
    dependency = json.loads((dependency_job/'proof.json').read_text())
    dependency_path = Path(dependency['dependency_path'])
    assert dependency_path.resolve().is_relative_to(dependency_job.resolve())
    for key in ('installed_only_into_new_owned_target', 'base_torch_triton_versions_unchanged',
                'native_SDK_and_models_unchanged', 'official_server_import_pass'):
        assert dependency[key], key
    assert dependency['server_source_SHA256'] == sha(package/'serve_gemma_runtime_columns.py')
    assert dependency['Gemma_source_preparation_SHA256'] == sha(receipt_path)
    for name, expected in dependency['files_SHA256'].items():
        f = dependency_path/name
        assert f.resolve().is_relative_to(dependency_path.resolve()) and sha(f) == expected, name
    artifacts, audits = {}, {}
    for arm in ('global', 'independent'):
        artifact = root/('full-'+arm+'-artifact')
        audit_path = artifact/'independent-full-artifact-audit.json'
        audit = json.loads(audit_path.read_text())
        validate_full_audit(audit, args.full_model_profile, arm == 'independent')
        assert sha(audit_path) == prep['full_audit_SHA256'][arm]
        assert audit['Gemma_source_preparation_SHA256'] == sha(receipt_path)
        assert audit['precision_kernel_sha256'] == sha(package/'gemma_runtime_precision.py')
        assert audit['bulk_prefill_A16_original_native_decode']
        export_gate_path = artifact.resolve().parent/'native-gate.json'
        export_gate = json.loads(export_gate_path.read_text())
        assert sha(export_gate_path) == audit['native_gate_sha256']
        assert export_gate['source'] == gate['source']
        for key in ('source_spec_SHA256', 'native_tests_xml_SHA256', 'unchanged_controls'):
            assert export_gate[key] == gate[key], key
        for name, expected in export_gate['binaries'].items():
            assert sha(name) == expected, name
        for name, record in audit['artifacts'].items():
            assert sha(artifact/name) == record['sha256'], name
        artifacts[arm], audits[arm] = artifact, audit
    binding = dict(actual_full_profile=args.full_model_profile,
        original_compiled_models_and_Python_source_reused=True,
        native_runtime_gate_SHA256=sha(args.full_native_job/'native-gate.json'),
        actual_sources_binaries_and_tests_reverified=True,
        model_source_preparation_SHA256=sha(receipt_path),
        current_native_source=str(native_source), model_Python_source=str(source),
        memory_worker_proof_SHA256=sha(args.memory_worker_proof),
        fresh_global_export_is_a_layout_control_not_the_previous_accepted_PTE=True,
        full_model_qualified=False, main_candidate_accepted=False)
    return dict(gate_path=gate_path, gate=gate, source=source, worker=worker,
                memory_proof=proof, dependency_path=dependency_path,
                Gemma_model_source_binding=binding, artifacts=artifacts,
                audits=audits, explicit_initial=True)
