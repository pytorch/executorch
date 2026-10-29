"""Relink the existing model worker with the paired-validation routing patch."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import traceback

from ready_open_group import patch


def main(args):
    rc = 1
    try:
        old = args.base/f'four-opt-{args.model}-combined-full-native-worker-20261028-v2'
        current = args.base/f'four-opt-{args.model}-observable-full-native-worker-20261028-v5'
        previous = json.loads((old/'build-commands.json').read_text())
        commands = json.loads((current/'build-commands.json').read_text())
        compile_command = previous[1][:]
        assert '-DNDEBUG' in compile_command
        assert any(x.startswith('-O') and x != '-O0' for x in compile_command)
        original = Path(compile_command[compile_command.index('-c')+1]).read_text()
        (args.job/'original-serving_runtime.cpp').open('x').write(original)
        (args.job/'candidate-serving_runtime.cpp').open('x').write(patch(original))
        compile_command[compile_command.index('-c')+1] = str(args.job/'candidate-serving_runtime.cpp')
        compile_command[compile_command.index('-o')+1] = str(args.job/'serving_runtime.o')
        link = commands[-1][:]
        old_object = str(old/'1-serving_runtime.o')
        assert link.count(old_object) == 1
        link[link.index(old_object)] = str(args.job/'serving_runtime.o')
        link[link.index('-o')+1] = str(args.job/'worker')
        (args.job/'build-commands.json').open('x').write(json.dumps([compile_command, link], indent=2))
        with (args.job/'build.log').open('x') as log:
            for command in [compile_command, link]:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        (args.job/'build-receipt.json').open('x').write(json.dumps(dict(
            model=args.model, worker=str(args.job/'worker'),
            worker_SHA256=hashlib.sha256((args.job/'worker').read_bytes()).hexdigest(),
            original_source_SHA256=hashlib.sha256(original.encode()).hexdigest(),
            candidate_source_SHA256=hashlib.sha256(patch(original).encode()).hexdigest(),
            kernel_and_other_objects_reused=True, numerical_qualification_pending=True), indent=2))
        rc = 0
    except BaseException:
        (args.job/'build-failure.txt').open('x').write(traceback.format_exc())
    finally:
        (args.job/'build-exit-receipt.json').open('x').write(json.dumps(dict(
            returncode=rc, UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())))
    return rc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--model', choices=['MG', 'Gemma'], required=True)
    raise SystemExit(main(parser.parse_args()))
