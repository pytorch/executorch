"""Bind a candidate-only HTTP diagnostic to an actual preserved global failure.

This does not waive candidate correctness, reclaim, or performance gates.
"""
import json
from pathlib import Path
from independent_session_full_native_binding import sha


def verify_failed_global_HTTP(job, pair):
    job, pair = Path(job).resolve(), Path(pair).resolve()
    state_path=job/'progress.json'
    failure=json.loads(state_path.read_text())
    assert failure['status']=='failed' and 'HTTP Error 500' in failure['error']
    handle=Path('/proc')/str(failure['controller_pid'])/'cmdline'
    assert not handle.exists() or b'run_independent_session_mini_http.py' not in handle.read_bytes()
    command_path=job.parent/'HTTP-memory-command.json'
    command=json.loads(command_path.read_text())['argv']
    assert '--full-model-profile' in command and command[command.index('--full-model-profile')+1]=='gemma4-48layers'
    assert '--arm' not in command or command[command.index('--arm')+1]=='both'
    old_pair=Path(command[command.index('--preparation-job')+1])
    for arm in ('global','independent'):
        name='full-'+arm+'-artifact/independent-full-artifact-audit.json'
        assert json.loads((old_pair/name).read_text())['artifacts']==json.loads((pair/name).read_text())['artifacts']
    assert not list(job.glob('round-*/off/independent')) and not list(job.glob('round-*/on/independent'))
    server=job/'round-1/off/global/server.log'
    text=server.read_text(errors='replace')
    assert '500 Internal Server Error' in text
    launch=job/'round-1/off/global/launch.json'
    launch_args=json.loads(launch.read_text())['argv']
    assert Path(launch_args[launch_args.index('--model-path')+1]).resolve()==(old_pair/'full-global-artifact/model.pte').resolve()
    paths=[state_path,job/'failure.json',command_path,server,launch,
           job/'round-1/off/global/requests.jsonl',job/'round-1/off/global/memory.jsonl']
    assert all(path.is_file() for path in paths)
    return dict(path=str(job),actual_failed_global_HTTP_preserved=True,
        same_global_and_independent_PTE_PTD=True,failed_global_test_is_not_a_pass=True,
        independent_arm_not_collected_in_failed_job=True,cause_not_yet_proven=True,
        raw_files_SHA256={str(path.relative_to(job)) if path.is_relative_to(job) else '../'+path.name:sha(path) for path in paths})
