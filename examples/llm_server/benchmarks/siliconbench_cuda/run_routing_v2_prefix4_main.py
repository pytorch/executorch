"""Run a full-model SiliconBench shard with routing v2 and four prefix entries."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback


def main(job):
    cfg = json.loads((job/'effective-config.json').read_text())
    completed = []; failed = []; rc = 1

    def progress(state, **extra):
        temp = job/'progress.pending.json'
        temp.write_text(json.dumps(dict(status=state, completed=completed, failed=failed,
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            numerical_qualification_pending=True, full_F1_done=False, **extra), indent=2))
        temp.replace(job/'progress.json')

    try:
        assert cfg['prefix_ON_entries'] == 4 and cfg['prefix_OFF_entries'] == 0
        assert cfg['active_sessions'] == cfg['decode_limit'] == 64
        assert cfg['initial_KV_capacity'] == 512 and cfg['logical_context'] == 131072
        assert cfg['prefill_chunk'] == (4096 if cfg['model'] == 'MG' else 2048)
        assert hashlib.sha256(Path(cfg['worker']).read_bytes()).hexdigest() == cfg['worker_SHA256']
        assert hashlib.sha256(Path(cfg['PTE']).read_bytes()).hexdigest() == cfg['PTE_SHA256']
        progress('verifying-same-full-weight-source')
        with Path(cfg['PTD']).open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == cfg['PTD_SHA256']
        env = dict(os.environ, ET_MAIN_CONFIG=str(job/'effective-config.json'), PYTHONDONTWRITEBYTECODE='1')
        for c in [1, 2, 4, 8, 16, 32, 64]:
            for prefix in ['off', 'on']:
                name = f"{cfg['model']}-ET-{cfg['split']}-{prefix}-c{c}"
                progress('collecting-full-main-performance', current_case=name)
                command = [cfg['HTTP_python'], '-B', str(job/'run_case_64_main.py'),
                    '--wrapper', str(job/'serve.sh'), '--backend', 'executorch',
                    '--model', cfg['model_id'], '--variant', 'routing-v2-prefix4',
                    '--prefix-cache', prefix, '--client', str(job/('benchmark_mg.py' if cfg['model']=='MG' else 'benchmark_gemma.py')),
                    '--prompts', str(job/(cfg['split']+'_benchmark_prompts.json')),
                    '--correctness-root', str(job/'deferred-correctness'),
                    '--python', cfg['HTTP_python'], '--concurrency', str(c),
                    '--port', str(cfg['port']), '--output', str(job/'matrix'/name),
                    '--lock', cfg['lock'], '--performance-rounds', '1', '--mode', 'performance', '--split', cfg['split']]
                with (job/(name+'.controller.log')).open('x') as log:
                    result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=9000)
                if result.returncode:
                    failed.append(dict(case=name, returncode=result.returncode))
                else:
                    completed.append(name)
        rc = 0 if len(completed) == 14 and not failed else 1
        progress('main-shard-complete-needs-independent-review', all_cases_collected=rc==0)
    except BaseException:
        (job/'failure.txt').open('x').write(traceback.format_exc())
        progress('main-shard-failed')
    finally:
        (job/'exit-receipt.json').open('x').write(json.dumps(dict(returncode=rc,
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), completed=len(completed),
            failed=failed, independent_review_pending=True, numerical_qualification_pending=True)))
    return rc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', type=Path, required=True)
    raise SystemExit(main(parser.parse_args().job))
