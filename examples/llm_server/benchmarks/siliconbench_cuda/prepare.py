"""Build a second stock llama.cpp configuration and pin the correctness inputs.

CPU preparation only. Existing framework source, model weights and experiment
records are preserved. Supervisor owns this process so SSH disconnects are safe.
"""
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
import urllib.request

import argparse
from collections import Counter

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root', type=Path, required=True, help='Task-owned model/venv directory')
parser.add_argument('--cpp-repo', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True, help='New preparation directory')
parser.add_argument('--python', default='python3')
parser.add_argument('--reuse-build', action='store_true')
parser.add_argument('--cpp-build-name', default='build-stock-native-evaluation')
parser.add_argument('--data-revision', help='Reuse the same pinned public dataset revision across machines')
args = parser.parse_args()
ROOT = args.root
GROUP = args.output
CPP = args.cpp_repo
GROUP.mkdir(parents=True, exist_ok=False)
EXPECTED_HEAD = '620d2225cc436ab0b001e3200e590e3b286ef9ab'


def state(status, **fields):
    q = GROUP / 'progress.pending.json'
    q.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                **fields), indent=2) + '\n')
    q.replace(GROUP / 'progress.json')


def run(args, timeout, log):
    with (GROUP / log).open('x') as out:
        subprocess.run(args, stdout=out, stderr=subprocess.STDOUT,
                       check=True, timeout=timeout)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(url):
    request = urllib.request.Request(url, headers={'User-Agent': 'SiliconBench-readonly-preparation'})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


try:
    state('building-stock-llamacpp-cublas-off', GPU_work_started=False)
    assert subprocess.check_output(['git', '-C', str(CPP), 'rev-parse', 'HEAD'],
                                   text=True).strip() == EXPECTED_HEAD
    assert not subprocess.check_output(['git', '-C', str(CPP), 'status',
                                        '--porcelain', '--untracked-files=no'], text=True).strip()
    cache = CPP / 'build-stock-cublas/CMakeCache.txt'
    options = {}
    for line in cache.read_text().splitlines():
        if not line or line.startswith(('#', '//')) or ':' not in line or '=' not in line:
            continue
        left, value = line.split('=', 1)
        key, kind = left.split(':', 1)
        if kind not in ['BOOL', 'STRING', 'UNINITIALIZED', 'FILEPATH']:
            continue
        if key.endswith('_FOUND') or value.endswith('-NOTFOUND'):
            continue
        if key.startswith(('GGML_', 'LLAMA_')) or key in [
            'BUILD_SHARED_LIBS', 'CMAKE_BUILD_TYPE', 'CMAKE_CUDA_ARCHITECTURES',
            'CMAKE_CUDA_COMPILER', 'CMAKE_CXX_COMPILER', 'CMAKE_C_COMPILER',
            'CMAKE_CXX_FLAGS_RELEASE', 'CMAKE_CUDA_FLAGS_RELEASE']:
            options[key] = value
    assert options['GGML_CUDA_FORCE_CUBLAS'] == 'ON'
    assert options['GGML_CUDA'] == 'ON' and options['CMAKE_CUDA_ARCHITECTURES'] == '120'
    original_options = dict(options)
    options['GGML_CUDA_FORCE_CUBLAS'] = 'OFF'
    build = CPP / args.cpp_build_name
    if args.reuse_build:
        assert build.exists() and (build/'bin/llama-server').exists()
    else:
        assert not build.exists(), 'Preserve an incomplete build; inspect before retrying'
        run(['cmake', '-S', str(CPP), '-B', str(build)] +
            [f'-D{k}={v}' for k, v in options.items()], 300, 'configure.log')
        run(['cmake', '--build', str(build), '--target', 'llama-server', '-j', '4'],
            3600, 'build.log')
    result_cache = (build / 'CMakeCache.txt').read_text()
    assert 'GGML_CUDA_FORCE_CUBLAS:BOOL=OFF' in result_cache
    assert 'GGML_CUDA_FORCE_MMQ:BOOL=OFF' in result_cache
    assert 'GGML_CUDA_GRAPHS:BOOL=ON' in result_cache
    binary = build / 'bin/llama-server'
    version = subprocess.run([str(binary), '--version'], capture_output=True,
                             text=True, check=True, timeout=45)
    assert not subprocess.check_output(['git', '-C', str(CPP), 'status',
                                        '--porcelain', '--untracked-files=no'], text=True).strip()
    proof = dict(stock_commit=EXPECTED_HEAD, stock_tracked_source_clean=True,
                 on_build_config_sha256=digest(cache), off_build_config_sha256=digest(build/'CMakeCache.txt'),
                 source_options=original_options, requested_options=options,
                 changed_requested_option={'GGML_CUDA_FORCE_CUBLAS': ['ON', 'OFF']},
                 existing_build_reused=args.reuse_build,
                 reused_build_all_other_options_require_independent_audit=args.reuse_build,
                 binary=str(binary), binary_sha256=digest(binary),
                 version=(version.stdout+version.stderr).strip(),
                 activation_policy='Stock CUDA routing; FORCE_CUBLAS OFF does not guarantee A16 bulk prefill',
                 CUDA_graphs=True, no_framework_code_patches=True)
    (GROUP / 'stock-cpp-off-build.json').write_text(json.dumps(proof, indent=2)+'\n')

    state('pinning-GMRID-correctness-data', stock_cpp_off_build_completed=True)
    api = 'https://api.github.com/repos/inflaton/llms-at-edge/commits/main'
    revision = args.data_revision or json.loads(fetch(api))['sha']
    assert len(revision) == 40 and all(c in '0123456789abcdef' for c in revision)
    data_dir = GROUP / 'correctness/data'
    data_dir.mkdir(parents=True)
    data_proof = dict(source_repository='inflaton/llms-at-edge', revision=revision, files={})
    for name in ['GMRID_v3-test.csv', 'categories.json']:
        url = f'https://raw.githubusercontent.com/inflaton/llms-at-edge/{revision}/dataset/{name}'
        content = fetch(url)
        (data_dir / name).write_bytes(content)
        data_proof['files'][name] = dict(url=url, sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    with (data_dir/'GMRID_v3-test.csv').open(newline='') as file:
        records = list(csv.DictReader(file))
    categories = json.loads((data_dir/'categories.json').read_text())
    assert len(records) == 1147 and len(categories) == 8
    assert len({str(row['id']) for row in records}) == 1147
    # Match the unmodified SiliconBench builder's input filtering. Preserve
    # the source-row audit independently of all runtime request failures.
    valid = []
    excluded = []
    for row in records:
        reasons = []
        if not row['Headline_Details'].strip():
            reasons.append('empty_input')
        if row['Summarized_label'].strip() not in categories:
            reasons.append('unknown_gold_label')
        if reasons:
            excluded.append(dict(id=str(row['id']), reasons=reasons,
                                 gold_label=row['Summarized_label'].strip()))
        else:
            valid.append(row)
    assert excluded == [dict(id='5079', reasons=['empty_input'], gold_label='Others')], excluded
    assert len(valid) == 1146
    valid_ids = {str(row['id']) for row in valid}
    data_proof.update(source_rows=len(records), evaluated_items=len(valid), excluded_source_rows=excluded,
                      filtering_policy='Unmodified SiliconBench build_prompts.py: skip empty input or unknown gold',
                      gold_label_distribution=dict(Counter(row['Summarized_label'].strip() for row in valid)))
    python = Path(args.python)
    scripts = Path(__file__).resolve().parent / 'correctness/scripts'
    (GROUP/'correctness/scripts').symlink_to(scripts, target_is_directory=True)
    for shots in [0, 5]:
        target = GROUP / f'correctness/prompts/{shots}shot'
        run([str(python), '-B', str(scripts/'build_prompts.py'), '--shots', str(shots),
             '--csv', str(data_dir/'GMRID_v3-test.csv'), '--categories', str(data_dir/'categories.json'),
             '--output-dir', str(target)], 60, f'build-prompts-{shots}.log')
        prompts = [json.loads(line) for line in (target/'prompts.jsonl').open()]
        labels = [json.loads(line) for line in (target/'labels.jsonl').open()]
        assert len(prompts) == len(labels) == len(valid)
        assert {str(row['id']) for row in prompts} == {str(row['id']) for row in labels} == valid_ids
        data_proof.setdefault('prompt_sets', {})[str(shots)] = dict(
            prompts_sha256=digest(target/'prompts.jsonl'), labels_sha256=digest(target/'labels.jsonl'),
            count=len(valid), builder_no_think_default=True, max_tokens=256)
    deps = subprocess.run([str(python), '-B', '-c', 'import aiohttp,sklearn; print(aiohttp.__version__,sklearn.__version__)'],
                          capture_output=True, text=True, timeout=60)
    if deps.returncode:
        # Install task evaluation dependencies only in our dedicated bench venv.
        pip = subprocess.run([str(python), '-m', 'pip', '--version'], capture_output=True,
                             text=True, timeout=30)
        if pip.returncode == 0:
            install = [str(python), '-m', 'pip', 'install']
        else:
            uv = shutil.which('uv')
            assert uv, 'Dedicated venv lacks pip; uv is required to install evaluation dependencies'
            install = [uv, 'pip', 'install', '--python', str(python)]
        run(install + ['aiohttp>=3.9', 'scikit-learn>=1.3'],
            600, 'correctness-dependencies.log')
        deps = subprocess.run([str(python), '-B', '-c', 'import aiohttp,sklearn; print(aiohttp.__version__,sklearn.__version__)'],
                              capture_output=True, text=True, check=True, timeout=60)
    data_proof['evaluation_dependency_versions'] = deps.stdout.strip()
    data_proof['scripts'] = {p.name: digest(p) for p in scripts.iterdir() if p.suffix == '.py'}
    data_proof['scorer_missing_response_policy'] = 'Verify every valid fixture response ID before scoring; never drop runtime errors or missing requests'
    (GROUP/'GMRID-input-audit.json').write_text(json.dumps(data_proof, indent=2)+'\n')
    state('completed', stock_cpp_off_build_completed=True, correctness_inputs_pinned=True,
          source_rows=len(records), excluded_source_rows=len(excluded),
          correctness_items=len(valid), correctness_shots=[0,5], no_GPU_work_started=True)
except Exception as exc:
    state('failed', error_type=type(exc).__name__, detail=str(exc)[-1200:])
    raise
