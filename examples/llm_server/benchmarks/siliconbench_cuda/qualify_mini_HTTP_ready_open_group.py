"""Pair current and queued-open HTTP routing on the identical saved MG mini."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import traceback

from ready_open_group import patch


def main(args):
    job, base = args.job, args.base
    rc = 1
    try:
        current = base/'four-opt-MG-observable-full-native-worker-20261028-v5'
        commands = json.loads((current/'build-commands.json').read_text())
        old = base/'four-opt-MG-combined-full-native-worker-20261028-v2'
        previous = json.loads((old/'build-commands.json').read_text())
        source_path = Path(previous[1][previous[1].index('-c')+1])
        source = source_path.read_text()
        (job/'original-serving_runtime.cpp').write_text(source)
        (job/'candidate-serving_runtime.cpp').write_text(patch(source))
        compile_command = previous[1][:]
        compile_command[compile_command.index('-c')+1] = str(job/'candidate-serving_runtime.cpp')
        compile_command[compile_command.index('-o')+1] = str(job/'serving_runtime.o')
        link = commands[-1][:]
        old_object = str(old/'1-serving_runtime.o')
        assert link.count(old_object) == 1
        link[link.index(old_object)] = str(job/'serving_runtime.o')
        link[link.index('-o')+1] = str(job/'worker')
        (job/'build-commands.json').write_text(json.dumps([compile_command, link], indent=2))
        with (job/'build.log').open('x') as log:
            for command in [compile_command, link]:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
        baseline = current/'worker'
        candidate = job/'worker'
        (job/'worker-bindings.json').write_text(json.dumps({arm:dict(path=str(path),
            SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
            for arm, path in [('baseline', baseline), ('candidate', candidate)]}, indent=2))
        harness = (job/'profile_mini_HTTP_routing_prefix_off.py').read_text()
        marker = "str(base/'four-opt-MG-observable-full-native-worker-20261028-v5/worker')"
        assert harness.count(marker) == 1
        for arm, worker in [('baseline', baseline), ('candidate', candidate)]:
            folder = job/arm
            folder.mkdir()
            script = folder/'HTTP-harness.py'
            script.write_text(harness.replace(marker, repr(str(worker))))
            command = [str(base/'venv/bin/python'), '-B', '-u', str(script),
                '--base', str(base), '--job', str(folder), '--port', str(args.port)]
            with (folder/'controller.log').open('x') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            assert json.loads((folder/'exit-receipt.json').read_text())['returncode'] == 0
        (job/'result.json').write_text(json.dumps(dict(mini_only=True, prefix_entries=0,
            paired_HTTP=True, main_unchanged=True, c=[1,2,4,8,16,32,64],
            four_rounds_discard_first=True, fresh_independent_review_required=True), indent=2))
        rc = 0
    except BaseException:
        (job/'failure.txt').write_text(traceback.format_exc())
    finally:
        (job/'exit-receipt.json').write_text(json.dumps(dict(returncode=rc,
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), mini_only=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--port', type=int, default=20273)
    main(parser.parse_args())
