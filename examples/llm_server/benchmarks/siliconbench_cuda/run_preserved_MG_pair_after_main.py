"""Compare preserved old/new MG on one GPU after a live main job finishes.

Use the original artifacts, native binaries and quantized weights. Reuse the
new runtime's full qualification only when every transported hash matches;
validate native CUDA and HTTP portability on the destination before timing.
This does not reproduce every historical Python dependency or accept results.
"""
import argparse
import collections
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while data := f.read(16 * 1024 * 1024):
            h.update(data)
    return h.hexdigest()


def review_smoke(case, correct, minimum_F1):
    """Recompute the complete final answers, not only protocol success."""
    from review_independent_full_task_quality import indexed, scorer_functions, final_object
    functions = scorer_functions(correct / 'scripts/score_f1.py')
    categories = json.loads((correct / 'data/categories.json').read_text())
    patterns = [(c, re.compile(r'\b' + re.escape(c.lower()) + r'\b')) for c in sorted(categories, key=len, reverse=True)]
    reviews = []
    for shot in (0, 5):
        inputs = case / ('smoke-inputs/' + str(shot) + 'shot')
        prompts, labels = indexed(inputs / 'prompts.jsonl'), indexed(inputs / 'labels.jsonl')
        canonical = indexed(correct / ('prompts/' + str(shot) + 'shot/prompts.jsonl'))
        responses_path = case / ('correctness/round-1/' + str(shot) + 'shot/responses.jsonl')
        responses = indexed(responses_path)
        assert len(responses) == 64 and set(responses) == set(labels) == set(prompts)
        pairs = []
        for key, row in responses.items():
            assert prompts[key] == canonical[key]
            assert not row.get('error') and row['finish_reason'] == 'stop'
            value, fenced = final_object(row.get('content'))
            assert isinstance(value, dict) and value.get('category') in categories
            cat, _ = functions['extract_category']((row.get('content') or '') + (row.get('reasoning') or ''), patterns)
            cat = functions['normalize_to_canonical'](cat, categories)
            assert cat == value['category'], 'Final-answer scoring differs from concatenated channels'
            request = row['request_parameters']
            assert request['temperature'] == 0 and request['top_p'] == 1 and request['max_tokens'] == 1024
            assert request['chat_template_kwargs'] == dict(enable_thinking=False, current_date='2026-10-21', reasoning_strength='low')
            pairs.append((labels[key]['gold_category'], cat))
        gold = collections.Counter(g for g, pred in pairs)
        pred = collections.Counter(pred for g, pred in pairs)
        tp = collections.Counter(g for g, pred in pairs if g == pred)
        weighted = sum(gold[c] * 2 * tp[c] / (gold[c] + pred[c]) if gold[c] + pred[c] else 0 for c in categories) / 64
        score_path = responses_path.with_name('scores.json')
        score = json.loads(score_path.read_text())
        assert abs(weighted - score['f1_weighted']) < 1e-10
        assert weighted >= minimum_F1, 'Portability smoke accuracy abnormal; preserve evidence and stop'
        reviews.append(dict(shot=shot, count=64, independently_recomputed_weighted_F1=weighted,
                            response_SHA256=sha(responses_path), labels_SHA256=sha(inputs / 'labels.jsonl'),
                            strict_final_JSON_complete=64, finish_stop=64, source_scores_SHA256=sha(score_path)))
    (case / 'independent-portability-quality-review.json').write_text(json.dumps(dict(
        shots=reviews, minimum_smoke_F1=minimum_F1, full_F1_not_claimed=True), indent=2) + '\n')


def binding(config):
    root = Path(config['bundle'])
    transfer = json.loads((root / 'transfer-provenance.json').read_text())
    assert transfer['every_file_SHA256_verified']
    manifest = transfer['manifest']
    assert manifest['kind'] == 'preserved-MG-old-new-runtime-pair'
    assert manifest['original_artifacts_runtime_and_weights_unmodified']
    for name, record in manifest['files'].items():
        path = root / name
        assert path.resolve().is_relative_to(root.resolve())
        assert path.stat().st_size == record['bytes'] and sha(path) == record['SHA256'], name
    qualified = json.loads((root / 'origin/full-qualification.json').read_text())
    assert qualified['full_profile'] == 'muse-glimmer-52layers'
    for key in ('full_model_correctness_pass', 'full_task_F1_independently_reviewed',
                'full_fixed_token_performance_pass', 'full_storage_rewind_graph_pass', 'full_HTTP_memory_pass'):
        assert qualified[key], key
    assert not qualified['unresolved_correctness_or_performance_signals']
    origin = json.loads((root / 'origin/source-revalidation.json').read_text())
    assert origin['all_origin_qualification_evidence_reverified']
    assert origin['qualification_SHA256'] == sha(root / 'origin/full-qualification.json')
    quality = json.loads((root / 'origin/full-task-raw-review.json').read_text())
    assert quality['complete'] and quality['actual_response_count'] == 9936 and quality['reviewed_shots'] == 20
    assert quality['full_task_raw_quality_pass'] and quality['full_task_F1_independently_reviewed']
    assert quality['artifact_SHA256'] == qualified['artifact_SHA256']
    for row in quality['shots']:
        assert row['errors'] == row['parse_failures'] == row['final_content_scoring_disagreements'] == 0
        assert row['finish_reasons'] == {'stop': row['count']}
    for name, expected in qualified['artifact_SHA256'].items():
        target = root / ('artifact/new/model.pte' if name == 'model.pte' else 'artifact/aoti_cuda_blob.ptd')
        assert sha(target) == expected
    source = Path(config['source'])
    for name, expected in origin['HTTP_source_files'].items():
        assert sha(source / name) == expected, name
    for name, expected in config['input_SHA256'].items():
        assert sha(Path(config['inputs']) / name) == expected, name
    package = Path(__file__).resolve().parent
    for name, expected in config['client_source_SHA256'].items():
        assert sha(package / name) == expected, name
    query = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,driver_version,power.limit', '--format=csv,noheader'], text=True).strip()
    assert query == config['expected_GPU_query'], query
    return root, qualified, query


def serve(a, config):
    root, qualified, query = binding(config)
    prefix = os.environ['SB_PREFIX_CACHE']
    assert prefix in ('off', 'on') and a.arm in ('old', 'new')
    worker = root / ('runtime/' + a.arm + '/worker')
    source = Path(config['source'])
    package = Path(__file__).resolve().parent
    entry = package / 'mg_preserved_error_logging_launcher.py' if a.arm == 'old' else source / 'examples/models/muse-glimmer/serving/serve_batching.py'
    argv = [config['server_python'], '-B', str(entry), '--worker-bin', str(worker),
            '--model-path', str(root / ('artifact/' + a.arm + '/model.pte')),
            '--data-path', str(root / 'artifact/aoti_cuda_blob.ptd'),
            '--tokenizer-path', str(root / 'tokenizer/tokenizer.json') if a.arm == 'old' else str(root / 'tokenizer'),
            '--hf-tokenizer', str(root / 'tokenizer'), '--model-id', 'muse-glimmer-30b',
            '--max-context', '131072', '--max-sessions', '16', '--max-decode-sequences', '16',
            '--max-inflight-requests', '16', '--prefix-cache-entries', '4' if prefix == 'on' else '0',
            '--prefix-checkpoint-interval', '0', '--host', '127.0.0.1', '--port', os.environ['SB_PORT'], '--backend', 'cuda']
    if a.arm == 'new':
        argv += ['--tool-parser', 'none']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(Path(config['dependency_path'])) + os.pathsep + str(source.parent),
               LD_LIBRARY_PATH=os.pathsep.join([str(root / ('runtime/' + a.arm)), *config['LD_library_paths']]))
    if a.arm == 'new':
        env['ET_CUDA_BENCH_MEMORY_TRACE'] = '1'
    else:
        env.pop('ET_CUDA_BENCH_MEMORY_TRACE', None)
    linked = subprocess.check_output(['ldd', str(worker)], env=env, text=True)
    assert 'not found' not in linked
    for name in ('libaoti_cuda_shims.so', 'libextension_cuda.so'):
        assert str(root / ('runtime/' + a.arm) / name) in linked, name
    print('PRESERVED_MG_PAIR_SERVER_BINDING:' + json.dumps(dict(
        argv=argv, arm=a.arm, GPU=query, worker_SHA256=sha(worker),
        PTE_SHA256=sha(root / ('artifact/' + a.arm + '/model.pte')),
        PTD_SHA256=qualified['artifact_SHA256']['aoti_cuda_blob.ptd'], prefix=prefix,
        origin_full_qualification_SHA256=sha(root / 'origin/full-qualification.json'),
        original_native_and_artifact_bytes_unchanged=True, exact_historical_Python_reproduction=False,
        origin_timings_not_used_as_target_timings=True, static_graphs_enabled=True,
        server_binding_monotonic_ns=time.monotonic_ns())), flush=True)
    os.execvpe(argv[0], argv, env)


def collect(a, config):
    a.output.mkdir(parents=True, exist_ok=False)
    def state(status, **extra):
        pending = a.output / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), accepted=False, **extra), indent=2) + '\n')
        pending.replace(a.output / 'progress.json')
    try:
        prior = Path(config['prior_main'])
        terminal = 'migration-native-HTTP-smoke-and-twenty-main-cases-collected'
        deadline = time.monotonic() + 36 * 3600
        while True:
            d = json.loads((prior / 'progress.json').read_text())
            if d['status'] == terminal and (prior / 'result.json').is_file():
                assert json.loads((prior / 'result.json').read_text())['main_cases_collected'] == 20
                break
            assert d['status'] != 'failed', d
            argv = (Path('/proc') / str(config['prior_main_PID']) / 'cmdline').read_bytes().split(b'\0')
            assert d['controller_pid'] == config['prior_main_PID']
            assert any(Path(x.decode()).name == 'run_migrated_Gemma_main.py' for x in argv if x)
            assert b'--output' in argv and argv[argv.index(b'--output') + 1] == os.fsencode(prior)
            assert time.monotonic() < deadline
            state('waiting-confirmed-live-Gemma-main', prior_PID=d['controller_pid'], prior_status=d['status'])
            time.sleep(20)
        root, qualified, query = binding(config)
        package = Path(__file__).resolve().parent
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip(), 'Preserve another GPU task'
        state('validating-identical-new-native-portability')
        env = dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join([str(root / 'runtime/new'), *config['LD_library_paths']]))
        xml = a.output / 'target-native-tests.xml'
        with Path(config['lock']).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            with (a.output / 'native-tests.log').open('x') as log:
                subprocess.run([str(root / 'runtime/new/test_cuda_kv_cache'), '--gtest_output=xml:' + str(xml)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        tests = list(ET.parse(xml).getroot().iter('testcase'))
        assert len(tests) == 74 and not any(any(t.find(k) is not None for k in ('error', 'failure', 'skipped')) for t in tests)
        (a.output / 'provenance.json').write_text(json.dumps(dict(config=config, config_SHA256=sha(a.config),
            target_GPU=query, target_native_tests=74, target_native_tests_SHA256=sha(xml),
            old_and_new_same_GPU_driver_power=True, source_artifacts_runtime_weights_unchanged=True,
            source_qualification_reused_not_recreated=True, natural_generation_lengths_may_differ=True,
            raw_review_and_fixed_token_causal_interpretation_pending=True), indent=2) + '\n')
        wrappers = {}
        for arm in ('old', 'new'):
            wrapper = a.output / ('serve-' + arm + '.sh')
            wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\ntest "${1:?}" = executorch\nexec ' + shlex.join([
                config['server_python'], '-B', str(Path(__file__).resolve()), '--phase', 'serve', '--arm', arm,
                '--config', str(a.config), '--output', str(a.output)]) + '\n')
            subprocess.run(['bash', '-n', str(wrapper)], check=True)
            wrappers[arm] = wrapper
        completed = []
        for mode in ('correctness', 'performance'):
            for split in (('agent',) if mode == 'correctness' else ('chat', 'agent')):
                for prefix in ('off', 'on'):
                    for c in ((16, 1) if mode == 'correctness' else (16, 1, 8, 4, 2)):
                        for arm in ('old', 'new'):
                            case = a.output / (arm + '-' + mode + '-' + split + '-' + prefix + '-c' + str(c))
                            argv = [config['benchmark_python'], '-B', str(package / 'run_case_24h.py'), '--wrapper', str(wrappers[arm]),
                                '--backend', 'executorch', '--model', 'muse-glimmer-30b', '--variant', 'preserved-' + arm + '-MG-same-GPU',
                                '--prefix-cache', prefix, '--client', str(package / 'benchmark_mg.py'),
                                '--prompts', str(Path(config['inputs']) / (split + '_benchmark_prompts.json')),
                                '--correctness-root', str(Path(config['inputs']) / 'correctness'), '--python', config['benchmark_python'],
                                '--concurrency', str(c), '--port', str(config['port']), '--output', str(case),
                                '--lock', config['lock'], '--mode', mode, '--split', split]
                            if mode == 'correctness':
                                argv += ['--quality-scope', 'smoke', '--correctness-rounds', '1', '--correctness-max-tokens', '1024',
                                    '--correctness-chat-template-kwargs', json.dumps(dict(enable_thinking=False, current_date='2026-10-21', reasoning_strength='low'))]
                            state('collecting-preserved-' + mode, arm=arm, case=str(case), completed=completed)
                            (a.output / (case.name + '-command.json')).write_text(json.dumps(dict(argv=argv), indent=2) + '\n')
                            with (a.output / (case.name + '.log')).open('x') as log:
                                subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=True, timeout=43200)
                            assert json.loads((case / 'progress.json').read_text())['status'] == 'completed'
                            if mode == 'correctness':
                                scores = json.loads((case / 'correctness-scores.json').read_text())
                                assert len(scores) == 2 and {s['shots'] for s in scores} == {0, 5}
                                assert all(s['n_scored'] == 64 and s['request_errors'] == 0 and s['n_missing_responses'] == s['n_extra_responses'] == 0 for s in scores)
                                review_smoke(case, Path(config['inputs']) / 'correctness', config['minimum_smoke_weighted_F1'])
                            else:
                                aggregate = json.loads((case / 'performance-aggregate.json').read_text())
                                assert aggregate['successful']['mean'] == 100 and aggregate['failed']['mean'] == 0
                            completed.append(case.name)
        assert len(completed) == 48
        (a.output / 'result.json').write_text(json.dumps(dict(completed=completed, performance_cells=40,
            portability_smoke_cells=8, accepted=False, independent_raw_review_pending=True,
            same_GPU_old_and_new_preserved_models=True), indent=2) + '\n')
        state('preserved-MG-pair-collected-pending-independent-review', completed=completed)
    except BaseException as error:
        state('failed', error=repr(error), traceback=traceback.format_exc())
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase', choices=('serve', 'collect'), required=True)
    p.add_argument('--arm', choices=('old', 'new'))
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    if a.phase == 'serve':
        serve(a, config)
    else:
        collect(a, config)


if __name__ == '__main__':
    main()
