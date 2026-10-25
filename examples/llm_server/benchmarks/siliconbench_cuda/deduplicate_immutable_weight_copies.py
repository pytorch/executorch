"""Share disk blocks for verified identical, immutable external weight copies.

Never edit weight bytes, a program, or an experiment record. Only explicitly
named duplicate files are replaced with hard links after recomputing hashes.
The retained source path stays in place; every model path remains readable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while b:=f.read(32*1024*1024):h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--destinations',type=Path,nargs='+',required=True)
    p.add_argument('--sha256',required=True);p.add_argument('--bytes',type=int,required=True)
    p.add_argument('--ledger',type=Path,required=True)
    a=p.parse_args();assert not a.ledger.exists()
    assert a.source.name=='aoti_cuda_blob.ptd' and not a.source.is_symlink()
    source_stat=a.source.stat();assert source_stat.st_size==a.bytes and sha(a.source)==a.sha256
    available_before=shutil.disk_usage(a.source.parent).free;records=[]
    for dest in a.destinations:
        assert dest!=a.source and dest.name=='aoti_cuda_blob.ptd' and not dest.is_symlink()
        before=dest.stat();assert before.st_dev==source_stat.st_dev and before.st_size==a.bytes
        assert (before.st_mode&0o777)==(source_stat.st_mode&0o777)
        assert sha(dest)==a.sha256,'Do not replace differing weights'
        if before.st_ino==source_stat.st_ino:
            records.append(dict(path=str(dest),already_shared=True));continue
        after_read=dest.stat();assert (before.st_ino,before.st_mtime_ns)==(after_read.st_ino,after_read.st_mtime_ns)
        tmp=dest.with_name(dest.name+'.verified-link-pending');assert not tmp.exists()
        os.link(a.source,tmp,follow_symlinks=False);os.replace(tmp,dest)
        current=dest.stat();assert current.st_ino==a.source.stat().st_ino
        assert current.st_size==a.bytes and sha(dest)==a.sha256
        records.append(dict(path=str(dest),old_inode=before.st_ino,new_inode=current.st_ino,
            bytes_unchanged=True,SHA256=a.sha256,previous_physical_bytes=before.st_blocks*512))
    assert sha(a.source)==a.sha256
    result=dict(source=str(a.source),source_SHA256=a.sha256,bytes_per_copy=a.bytes,
        files=records,free_before=available_before,free_after=shutil.disk_usage(a.source.parent).free,
        every_model_path_preserved=True,no_weight_bytes_modified=True,
        immutable_weights_only=True,program_and_experiment_files_unchanged=True)
    a.ledger.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
