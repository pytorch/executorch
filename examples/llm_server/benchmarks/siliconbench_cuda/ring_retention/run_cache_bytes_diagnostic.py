"""Compare logical cached K/V and dense selected logits after each cold step."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        while b := f.read(16 << 20): h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('native-build', 'export-job', 'source-scripts', 'output', 'lock'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.source_scripts))
    from run_real_mini_four_arms import Probe, token, logits, compare
    import numpy as np
    proof = json.loads((args.native_build / 'proof.json').read_text())
    binary = Path(proof['probe'])
    assert digest(binary) == proof['probe_SHA256']
    audits = {a: json.loads((args.export_job / a / 'artifact-audit.json').read_text())
              for a in ('baseline', 'ring')}
    assert audits['baseline']['artifacts']['aoti_cuda_blob.ptd']['SHA256'] == audits['ring']['artifacts']['aoti_cuda_blob.ptd']['SHA256']
    for audit in audits.values():
        for name, value in audit['artifacts'].items():
            assert digest(Path(audit['artifact']) / name) == value['SHA256']
    records = {a: [] for a in audits}
    vectors = {a: {} for a in audits}
    caches = {a: {} for a in audits}
    with args.lock.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
            '--format=csv,noheader'], text=True).strip()
        for arm, audit in audits.items():
            probe = Probe(binary, Path(audit['artifact']), args.output / arm, False)
            try:
                for concurrency in (1, 4):
                    owners = [probe.request(dict(op='new'))['seq_id'] for _ in range(concurrency)]
                    for cycle in range(16):
                        tokens = [token(i, cycle) for i in range(concurrency)]
                        reply = probe.request(dict(op='step', seq_ids=owners, tokens=tokens,
                            select=list(range(concurrency)), argmax=False, save_logits=True))
                        value = logits(reply)
                        assert np.isfinite(value).all()
                        vectors[arm][concurrency, cycle] = value
                        dumps = []
                        for owner in owners:
                            row = probe.request(dict(op='dump_kv', seq_id=owner,
                                directory=f'kv-c{concurrency}-step{cycle}-sid{owner}'))
                            directory = Path(row['directory'])
                            meta = json.loads((directory / 'metadata.json').read_text())
                            files = []
                            for layer in meta['layers']:
                                shape = (meta['length'], layer['kv_heads'], layer['head_dim'])
                                for kind in ('K', 'V'):
                                    path = directory / f'layer-{layer["layer"]}-{kind}.bf16'
                                    bits = np.fromfile(path, dtype=np.uint16).reshape(shape)
                                    caches[arm][concurrency, cycle, owner, layer['layer'], kind] = bits
                                    files.append(dict(file=str(path), SHA256=digest(path), shape=shape))
                            dumps.append(dict(metadata=meta, files=files))
                        records[arm].append(dict(concurrency=concurrency, cycle=cycle,
                            input_tokens=tokens, input_owners=owners, reply=reply, caches=dumps,
                            logit_files=[dict(file=r['file'], SHA256=digest(r['file'])) for r in reply['logits']]))
                    for owner in owners: probe.request(dict(op='remove', seq_id=owner))
            finally: probe.close()
            (args.output / (arm + '-records.json')).write_text(json.dumps(records[arm], indent=2) + '\n')
    numerical, cached = [], []
    for old, new in zip(records['baseline'], records['ring']):
        assert old['input_tokens'] == new['input_tokens'] and old['input_owners'] == new['input_owners']
        c, step = old['concurrency'], old['cycle']
        numerical.append(dict(concurrency=c, cycle=step,
            **compare(vectors['baseline'][c, step], vectors['ring'][c, step])))
    for key, old in caches['baseline'].items():
        new = caches['ring'][key]
        as_float = lambda v: (v.astype(np.uint32) << 16).view(np.float32)
        cached.append(dict(concurrency=key[0], cycle=key[1], owner=key[2], layer=key[3], kind=key[4],
            byte_exact=bool(np.array_equal(old, new)), different_elements=int(np.count_nonzero(old != new)),
            total_elements=int(old.size), **compare(as_float(old), as_float(new))))
    result = dict(numerics=numerical, cached_logical_KV=cached, native_build_proof=proof,
        artifact_audits=audits, diagnostic_only=True, device_argmax_enabled=False,
        no_reexport=True, no_wrap_or_growth=True, mini_qualified=False,
        first_divergent_step={str(c): next((r['cycle'] for r in numerical
            if r['concurrency'] == c and r['normalized_RMS'] != 0), None) for c in (1, 4)})
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(result_SHA256=digest(args.output / 'result.json'),
                         first_divergent_step=result['first_divergent_step'])))


if __name__ == '__main__': main()
