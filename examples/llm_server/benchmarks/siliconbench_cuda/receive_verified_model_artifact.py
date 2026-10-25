"""Receive an exact previously qualified public-GGUF artifact into a new folder.

Validate every byte against the supplied source hashes. Refuse links, extra
files, path traversal or replacement of an existing destination. Incomplete
transfers remain in their separate partial directory for inspection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    manifest=json.loads(a.manifest.read_text());expected=manifest['files']
    assert manifest['public_GGUF_source_verified']
    assert set(expected)=={'model.pte','aoti_cuda_blob.ptd','artifact-audit.json','sampler-policy.json'}
    assert not a.destination.exists()
    partial=a.destination.with_name(a.destination.name+'.partial');partial.mkdir(parents=True,exist_ok=False)
    total=sum(r['bytes'] for r in expected.values())
    assert shutil.disk_usage(partial).free>total+4*1024**3,'Leave staging room for active jobs'
    seen=set()
    with tarfile.open(fileobj=sys.stdin.buffer,mode='r|*') as archive:
        for member in archive:
            name=member.name
            assert member.isfile() and name in expected and name not in seen
            assert Path(name).name==name and member.size==expected[name]['bytes']
            h=hashlib.sha256();count=0
            with archive.extractfile(member) as source,(partial/name).open('xb') as output:
                while b:=source.read(16*1024*1024):h.update(b);output.write(b);count+=len(b)
            assert count==member.size and h.hexdigest()==expected[name]['SHA256']
            seen.add(name);print(json.dumps(dict(file=name,bytes=count,SHA256=h.hexdigest(),verified=True)),flush=True)
    assert seen==set(expected)
    (partial/'transfer-provenance.json').write_text(json.dumps(dict(source_manifest=manifest,
        source_manifest_SHA256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
        every_received_file_SHA256_verified=True,source_unchanged=True),indent=2)+'\n')
    partial.rename(a.destination)
    print('TASK_JSON:'+json.dumps(dict(destination=str(a.destination),bytes=total,
        all_source_hashes_verified=True,existing_model_paths_unmodified=True)),flush=True)


if __name__=='__main__':main()
