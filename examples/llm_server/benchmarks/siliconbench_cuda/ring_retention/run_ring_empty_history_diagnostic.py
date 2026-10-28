"""Locate ring logit divergence with existing mini artifacts, without re-export."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(16 << 20):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-scripts', 'native-preparation', 'export-job', 'output', 'lock'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    progress = args.output / 'progress.json'

    def state(status, **fields):
        pending = progress.with_suffix('.pending.json')
        pending.write_text(json.dumps(dict(status=status, PID=os.getpid(),
            qualification=False, full_models_tested=False, **fields)) + '\n')
        pending.replace(progress)

    try:
        state('verifying-existing-artifacts')
        native = json.loads((args.native_preparation / 'native-gate.json').read_text())
        binary = Path(native['probe'])
        assert digest(binary) == native['probe_SHA256']
        audits = {a: json.loads((args.export_job / a / 'artifact-audit.json').read_text())
                  for a in ('baseline', 'ring')}
        assert audits['baseline']['model'] == audits['ring']['model'] == 'MG'
        assert audits['baseline']['artifacts']['aoti_cuda_blob.ptd']['SHA256'] == audits['ring']['artifacts']['aoti_cuda_blob.ptd']['SHA256']
        for a in audits.values():
            for name, record in a['artifacts'].items():
                assert digest(Path(a['artifact']) / name) == record['SHA256']
        sys.path.insert(0, str(args.source_scripts))
        from run_real_mini_four_arms import Probe, token, logits, compare
        import numpy as np

        # These are teacher-forced numerical probes, not an argmax/lifecycle
        # acceptance. The ordinary argmax gate remains a separate requirement.
        def step(probe, owners, tokens, save):
            selected = [i for i in range(len(owners))
                        if i == len(owners) - 1 or owners[i] != owners[i + 1]]
            return probe.request(dict(op='step', seq_ids=owners, tokens=tokens,
                select=selected, argmax=False, save_logits=save))

        rows = {a: [] for a in audits}
        arrays = {a: {} for a in audits}
        with args.lock.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                '--format=csv,noheader'], text=True).strip()
            for arm, audit in audits.items():
                state('running-numerical-diagnostic', arm=arm)
                probe = Probe(binary, Path(audit['artifact']), args.output / arm, False)
                try:
                    for history in (0, 1, 64, 1024):
                        for label, widths in [('d1', [1]), ('d4', [1] * 4),
                                              ('d16', [1] * 16), ('p64', [64]), ('p512', [512])]:
                            owners = [probe.request(dict(op='new'))['seq_id'] for _ in widths]
                            for i, owner in enumerate(owners):
                                if history:
                                    step(probe, [owner] * history,
                                         [token(i, q) for q in range(history)], False)
                            positions = [history] * len(owners)
                            for cycle in range(16):
                                ids, tokens = [], []
                                for i, (owner, width) in enumerate(zip(owners, widths)):
                                    ids += [owner] * width
                                    tokens += [token(i, q) for q in range(positions[i], positions[i] + width)]
                                    positions[i] += width
                                save = cycle in (0, 15)
                                reply = step(probe, ids, tokens, save)
                                if save:
                                    key = (history, label, cycle)
                                    value = logits(reply)
                                    assert np.isfinite(value).all()
                                    arrays[arm][key] = value
                                    rows[arm].append(dict(history=history, case=label, cycle=cycle,
                                        input_tokens=tokens, input_owners=ids, reply=reply,
                                        logit_files=[dict(file=r['file'], SHA256=digest(r['file'])) for r in reply['logits']]))
                            for owner in owners:
                                probe.request(dict(op='remove', seq_id=owner))
                finally:
                    probe.close()
                (args.output / (arm + '-records.json')).write_text(json.dumps(rows[arm], indent=2) + '\n')
        numerical = []
        for old, new in zip(rows['baseline'], rows['ring']):
            assert old['input_tokens'] == new['input_tokens'] and old['input_owners'] == new['input_owners']
            key = (old['history'], old['case'], old['cycle'])
            numerical.append(dict(history=key[0], case=key[1], cycle=key[2],
                **compare(arrays['baseline'][key], arrays['ring'][key])))
        result = dict(numerics=numerical, artifact_audits=audits,
            native_gate=native, diagnostic_only=True, device_argmax_enabled=False,
            model_qualified=False, full_models_tested=False,
            source_script_SHA256=digest(args.source_scripts / 'run_real_mini_four_arms.py'))
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        state('complete-diagnostic-review-required', result_SHA256=digest(args.output / 'result.json'))
    except BaseException:
        state('failed-preserve-evidence', traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
