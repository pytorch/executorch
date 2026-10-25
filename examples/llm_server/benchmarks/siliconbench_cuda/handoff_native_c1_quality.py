"""Retain one native full cell and split the other c1 cells without duplicates.

Only the verified CPU queue parent is held. Its current case and inference
continue. The peer may start only after the original parent is terminal.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

from enforce_owned_tuning_deadline import identity, stop_owned_tree


class ParentHold:
    def __init__(self, pid, expected):
        self.pid = pid
        self.expected = tuple(expected)
        self.held = False

    def __enter__(self):
        assert identity(self.pid) == self.expected
        os.kill(self.pid, signal.SIGSTOP)
        self.held = True
        return self

    def __exit__(self, *unused):
        if self.held and identity(self.pid) == self.expected:
            os.kill(self.pid, signal.SIGCONT)
        self.held = False

    def stop_after_child_finished(self):
        assert self.held and identity(self.pid) == self.expected
        os.kill(self.pid, signal.SIGTERM)
        os.kill(self.pid, signal.SIGCONT)
        self.held = False
        end = time.monotonic() + 90
        while identity(self.pid) is not None:
            assert time.monotonic() < end, 'Verified CPU queue did not terminate'
            time.sleep(.1)


def interrupted(signum, frame):
    raise InterruptedError('Owned handoff interrupted; restore held CPU parent')


def self_test():
    """Exercise real Linux signal restoration while an independent child runs."""
    with tempfile.TemporaryDirectory(prefix='native-c1-hold-test-') as temp:
        marker = Path(temp) / 'child-count'
        child_code = ('import pathlib,time; p=pathlib.Path(' + repr(str(marker))
                      + '); n=0\nwhile True:\n n+=1;p.write_text(str(n));time.sleep(.05)')
        child = subprocess.Popen([sys.executable, '-c', child_code], start_new_session=True)
        parent = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(60)'])
        helper = None
        expected = identity(parent.pid)
        try:
            assert expected and identity(child.pid)
            code = ('import sys,time,signal;sys.path.insert(0,'
                    + repr(str(Path(__file__).resolve().parent)) + ');'
                    'from handoff_native_c1_quality import ParentHold,interrupted;'
                    'signal.signal(signal.SIGTERM,interrupted)\n'
                    'try:\n with ParentHold(' + str(parent.pid) + ',' + repr(expected)
                    + '):\n  print("held",flush=True);time.sleep(60)\n'
                    'except InterruptedError:\n print("restored",flush=True)')
            helper = subprocess.Popen([sys.executable, '-B', '-c', code],
                                      stdout=subprocess.PIPE, text=True)
            assert helper.stdout.readline().strip() == 'held'
            assert Path('/proc', str(parent.pid), 'stat').read_text().rsplit(')', 1)[1].split()[0] == 'T'
            end = time.monotonic() + 3
            while not marker.exists():
                assert time.monotonic() < end
                time.sleep(.05)
            before = int(marker.read_text())
            time.sleep(.2)
            after = int(marker.read_text())
            assert after > before and identity(child.pid)
            helper.send_signal(signal.SIGTERM)
            assert helper.stdout.readline().strip() == 'restored'
            assert helper.wait(timeout=5) == 0 and identity(parent.pid) == expected
            assert Path('/proc', str(parent.pid), 'stat').read_text().rsplit(')', 1)[1].split()[0] != 'T'
            return dict(passed=True,external_SIGTERM_restores_verified_CPU_parent=True,
                        independent_child_continued_while_parent_held=True,
                        parent_start_identity_unchanged=True)
        finally:
            for process in [helper, parent, child]:
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                    process.wait(timeout=5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--matrix', type=Path)
    p.add_argument('--controller-pid', type=int)
    p.add_argument('--controller-start', type=int)
    p.add_argument('--controller-command-sha256')
    p.add_argument('--own-case-plan', type=Path)
    p.add_argument('--peer-case-plan', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--timeout', type=int, default=14400)
    p.add_argument('command', nargs=argparse.REMAINDER)
    a = p.parse_args()
    if a.self_test:
        print('TASK_JSON:' + json.dumps(self_test()))
        return
    assert all(x is not None for x in [a.matrix, a.controller_pid, a.controller_start,
                                     a.controller_command_sha256, a.own_case_plan,
                                     a.peer_case_plan, a.output])
    assert 1 <= a.timeout <= 14400 and not a.output.exists()
    command = a.command[1:] if a.command[:1] == ['--'] else a.command
    assert command and Path(command[0]).is_absolute()
    active = 'correctness-agent-vllm-prefix-off-c1'
    original = json.loads((a.matrix / 'matrix.json').read_text())
    state = json.loads((a.matrix / 'progress.json').read_text())
    own = json.loads(a.own_case_plan.read_text())
    peer = json.loads(a.peer_case_plan.read_text())
    assert state['controller_pid'] == a.controller_pid and state['current_case'] == active
    assert not state['completed']
    assert original['model'] == own['model'] == peer['model'] == 'muse-glimmer-30b'
    original_set = {tuple(row) for row in original['cases']}
    retained = {('correctness', 'off', 1, 'vllm')}
    own_set, peer_set = {tuple(row) for row in own['cases']}, {tuple(row) for row in peer['cases']}
    assert original_set == {('correctness', cache, 1, backend)
                            for cache in ['off', 'on'] for backend in ['vllm', 'sglang']}
    assert own_set == {('correctness', 'on', 1, 'vllm')}
    assert peer_set == {('correctness', cache, 1, 'sglang') for cache in ['off', 'on']}
    assert not (retained & own_set or retained & peer_set or own_set & peer_set)
    assert retained | own_set | peer_set == original_set
    expected = (a.controller_pid, a.controller_start)
    raw = Path('/proc', str(a.controller_pid), 'cmdline').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == a.controller_command_sha256
    assert b'/run_matrix_24h.py\x00' in raw and str(a.matrix).encode() in raw
    a.output.mkdir(parents=True)
    (a.output / 'original-CPU-queue-state.json').write_text(json.dumps(state, indent=2) + '\n')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def save(status, **fields):
        pending = a.output / 'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                          retained_case=active, previous_controller=a.controller_pid,
                                          **fields), indent=2) + '\n')
        pending.replace(a.output / 'progress.json')

    parent_stopped = False
    child = None
    try:
        with ParentHold(a.controller_pid, expected) as hold:
            save('waiting-retained-full-native-cell', active_inference_not_stopped=True,
                 own_remaining_cells=1, peer_remaining_cells=2,
                 peer_must_wait_original_parent_terminal=True)
            current = a.matrix / active
            end = time.monotonic() + a.timeout
            while True:
                assert time.monotonic() < end, 'Bounded native handoff expired'
                pr = json.loads((current / 'progress.json').read_text())
                assert pr['status'] not in ['failed', 'cancelled', 'aborted']
                if pr['status'] == 'completed' and identity(pr['controller_pid']) is None:
                    break
                assert identity(a.controller_pid) == expected
                time.sleep(5)
            settings = json.loads((current / 'settings.json').read_text())
            assert settings['mode'] == 'correctness' and settings['concurrency'] == 1
            assert settings['variant'] == 'vllm' and settings['quality_scope'] == 'full'
            assert settings['correctness_items'] == 1146 and settings['correctness_rounds'] == 1
            assert (current / 'memory-summary.json').exists()
            hashes = {}
            for shot in [0, 5]:
                target = current / f'correctness/round-1/{shot}shot'
                rows = [json.loads(line) for line in (target / 'responses.jsonl').read_text().splitlines()]
                score = json.loads((target / 'scores.json').read_text())
                assert len(rows) == len({str(row['id']) for row in rows}) == 1146
                assert score['n_scored'] == 1146 and score['n_missing_responses'] == score['n_extra_responses'] == 0
                assert score['request_errors'] == 0
                for path in [target / 'responses.jsonl', target / 'scores.json']:
                    hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            hold.stop_after_child_finished()
            parent_stopped = True
            assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == sha for path, sha in hashes.items())
            (a.matrix / 'retirement-disjoint-correctness-handoff.json').write_text(
                json.dumps(dict(original_CPU_parent_terminal=True, retained_case=active,
                                retained_response_score_SHA256=hashes, retained_complete_cell_preserved=True,
                                own_plan=str(a.own_case_plan), peer_plan=str(a.peer_case_plan)), indent=2) + '\n')
            save('original-CPU-parent-terminal', original_complete_cell_preserved=True,
                 own_remaining_cells=1, peer_remaining_cells=2)
            child = subprocess.Popen(command)
            save('running-own-continuation', original_CPU_parent_terminal=True,
                 continuation_pid=child.pid, peer_remaining_cells=2)
            assert child.wait() == 0
            save('completed', original_CPU_parent_terminal=True,
                 retained_complete_cells=1, own_complete_cells=1, peer_assigned_cells=2,
                 primary_scope_unchanged=True)
    except BaseException as exc:
        if child is not None and identity(child.pid) is not None:
            item = identity(child.pid)
            stop_owned_tree(child.pid, lambda: os.kill(item[0], signal.SIGKILL)
                            if identity(item[0]) == item else None)
            child.wait(timeout=10)
        save('failed', error_type=type(exc).__name__, detail=str(exc)[-1000:],
             original_CPU_parent_terminal=parent_stopped,
             held_parent_restored=not parent_stopped)
        raise


if __name__ == '__main__':
    main()
