"""Detached F1 and fixed-sample validation of the exact current main runtime."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import subprocess
import sys
import traceback


def main():
    job = Path(sys.argv[1])
    cfg = json.loads((job / 'effective-config.json').read_text())
    tasks = json.loads((job / 'tasks.json').read_text())
    completed, failed = [], []
    rc = 1

    def status(state, **fields):
        value = dict(status=state, completed=completed, failed=failed,
                     UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), **fields)
        temporary = job / 'progress.pending.json'
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(job / 'progress.json')

    try:
        status('preflight')
        assert cfg['prefix_ON_entries'] == 4 and cfg['prefix_OFF_entries'] == 0
        assert cfg['active_sessions'] == cfg['decode_limit'] == 64
        assert hashlib.sha256(Path(cfg['worker']).read_bytes()).hexdigest() == cfg['worker_SHA256']
        assert all(Path(cfg[key]).is_file() for key in ['PTE', 'PTD'])
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                             '--format=csv,noheader'], text=True).strip()
        manifest = json.loads((job / 'source-and-input-manifest.json').read_text())
        for name, sha in manifest['files'].items():
            assert hashlib.sha256((job / name).read_bytes()).hexdigest() == sha, name
        env = dict(os.environ, ET_MAIN_CONFIG=str(job / 'effective-config.json'),
                   PYTHONDONTWRITEBYTECODE='1')
        for task in tasks:
            scope, concurrency, prefix = task
            name = f"{cfg['model']}-{scope}-{prefix}-c{concurrency}"
            status('running', current=name)
            argv = [cfg['HTTP_python'], '-B', str(job / 'run_case_64_main.py'),
                    '--wrapper', str(job / 'serve.sh'), '--backend', 'executorch',
                    '--model', cfg['model_id'], '--variant', 'routing-v2-prefix4-current-main-quality',
                    '--prefix-cache', prefix, '--client', str(job / 'benchmark_mg.py'),
                    '--prompts', str(job / 'agent_benchmark_prompts.json'),
                    '--correctness-root', str(job / 'evaluation/correctness'),
                    '--python', str(job / 'evaluation/.venv/bin/python'),
                    '--concurrency', str(concurrency), '--port', str(cfg['port']),
                    '--output', str(job / 'cases' / name), '--lock', cfg['lock'],
                    '--performance-rounds', '1', '--correctness-rounds', '1',
                    '--mode', 'correctness', '--split', 'agent', '--quality-scope', scope]
            if cfg['model'] == 'MG':
                argv += ['--correctness-max-tokens', '1024', '--correctness-chat-template-kwargs',
                         json.dumps(dict(enable_thinking=False, reasoning_strength='low', current_date='2026-10-21'))]
            else:
                argv += ['--correctness-max-tokens', '256']
            with (job / (name + '.controller.log')).open('x') as log:
                result = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=86400)
            (completed if result.returncode == 0 else failed).append(name)
            status('case-finished', current=name, returncode=result.returncode)
        rc = 1 if failed else 0
        status('complete-needs-independent-review')
    except BaseException:
        (job / 'failure.txt').write_text(traceback.format_exc())
        status('failed')
    finally:
        (job / 'exit-receipt.json').write_text(json.dumps(dict(returncode=rc, completed=completed,
            failed=failed, UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()), indent=2))
    sys.exit(rc)


if __name__ == '__main__':
    main()
