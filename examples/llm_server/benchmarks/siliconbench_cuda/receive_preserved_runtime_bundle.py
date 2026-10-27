"""Receive hash-bound artifacts and native binaries into a new owned directory.

No received file is executed. Source code is deployed separately. Preserve a
failed partial transfer; never replace an existing destination or other data.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args()
    manifest = json.loads(a.manifest.read_text())
    assert manifest['kind'] == 'preserved-MG-old-new-runtime-pair'
    assert manifest['public_GGUF_source_verified']
    assert manifest['original_artifacts_runtime_and_weights_unmodified']
    assert not a.destination.exists()
    partial = a.destination.with_name(a.destination.name + '.partial')
    partial.mkdir(parents=True, exist_ok=False)
    expected = manifest['files']
    assert shutil.disk_usage(partial).free > sum(r['bytes'] for r in expected.values()) + 4 * 1024**3
    seen = set()
    with tarfile.open(fileobj=sys.stdin.buffer, mode='r|*') as archive:
        for member in archive:
            name = member.name
            path = Path(name)
            assert member.isfile() and name in expected and name not in seen
            assert not path.is_absolute() and '..' not in path.parts
            assert path.parts[0] in ('artifact', 'runtime', 'origin', 'tokenizer')
            assert path.suffix not in ('.py', '.sh'), 'Deploy source through its separate source channel'
            record = expected[name]
            assert member.size == record['bytes']
            target = partial / path
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with target.open('xb') as out, archive.extractfile(member) as incoming:
                while data := incoming.read(16 * 1024 * 1024):
                    h.update(data)
                    out.write(data)
            assert h.hexdigest() == record['SHA256'], name
            target.chmod(record['mode'])
            seen.add(name)
            print(json.dumps(dict(file=name, bytes=member.size, SHA256=h.hexdigest(), verified=True)), flush=True)
    assert seen == set(expected)
    (partial / 'transfer-provenance.json').write_text(json.dumps(dict(
        manifest=manifest, manifest_SHA256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        every_file_SHA256_verified=True, original_files_unmodified=True), indent=2) + '\n')
    partial.rename(a.destination)
    print('TASK_JSON:' + json.dumps(dict(destination=str(a.destination), complete=True)), flush=True)


if __name__ == '__main__':
    main()
