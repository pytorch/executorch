"""Recover a failed stock llama-server launch from a noexec scratch mount.

Only the named experiment output and its sibling recovery directory are used.
The existing build, GGUF files, source and failure evidence remain read only.
Copied executable bytes must match the already audited stock binary.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(16*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('existing-job','output','existing-binary-dir'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args()
    old=args.existing_job.resolve();job=args.output.resolve();original=args.existing_binary_dir.resolve()
    assert str(old).startswith('/workspace/siliconbench-')
    assert job.parent==old.parent and job.name.startswith(old.name+'-') and not job.exists()
    assert str(original).startswith('/dev/shm/siliconbench-') and original.name=='bin'
    assert json.loads((old/'progress.json').read_text())['status']=='failed-preserve-existing-results'
    assert json.loads((old/'matrix/progress.json').read_text())['completed']==[]
    cfg=json.loads((old/'matrix-config.json').read_text())
    assert len(cfg['tasks'])==2 and all(t['backend']=='llamacpp' for t in cfg['tasks'])
    assert all(t['CPP_commit']=='620d2225cc436ab0b001e3200e590e3b286ef9ab' for t in cfg['tasks'])
    assert sha(original/'llama-server')==cfg['tasks'][0]['CPP_binary_SHA256']
    assert shutil.disk_usage(job.parent).free>sum(p.stat().st_size for p in original.iterdir() if p.is_file())+1024**3
    job.mkdir()
    binary=job/'stock-bin';shutil.copytree(original,binary,symlinks=True)
    hashes={}
    for path in original.iterdir():
        if path.is_file():
            digest=sha(path);assert sha(binary/path.name)==digest;hashes[path.name]=digest
    (job/'identical-stock-bin-proof.json').write_text(json.dumps(dict(original=str(original),
        executable_copy=str(binary),files_SHA256=hashes,no_framework_code_change=True,no_GGUF_change=True),indent=2)+'\n')
    for task in cfg['tasks']:
        before=Path(task['wrapper']);wrapper=job/before.name
        text=before.read_text();assert str(original/'llama-server') in text
        text=text.replace(str(original/'llama-server'),str(binary/'llama-server'))
        assert text.count('\nexec ')==1
        text=text.replace('\nexec ','\nexport LD_LIBRARY_PATH='+shlex.quote(str(binary))+
            '"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\nexec ',1)
        wrapper.write_text(text)
        subprocess.run(['bash','-n',str(wrapper)],check=True)
        task['wrapper']=str(wrapper)
    (job/'inputs').symlink_to(old/'inputs',target_is_directory=True)
    config=job/'matrix-config.json';config.write_text(json.dumps(cfg,indent=2)+'\n')
    env=dict(os.environ,LD_LIBRARY_PATH=str(binary)+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''))
    linked=subprocess.check_output(['ldd',str(binary/'llama-server')],env=env,text=True)
    assert str(original) not in linked and 'not found' not in linked
    (job/'copied-library-resolution.log').write_text(linked)
    package=Path(__file__).resolve().parent
    command=[sys.executable,'-B','-u',str(package/'run_default_chunk_single_matrix.py'),
        '--config',str(config),'--output',str(job/'matrix')]
    quality=job.with_name(job.name+'-full-F1')
    qcommand=[sys.executable,'-B','-u',str(package/'run_default_chunk_deferred_native_full_quality.py'),
        '--native-job',str(job),'--output',str(quality)]
    with quality.with_name(quality.name+'-controller.log').open('x') as log:
        q=subprocess.Popen(qcommand,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (job/'launch.json').write_text(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        matrix_argv=command,quality_PID=q.pid,quality_argv=qcommand,previous_failure_retained=str(old)),indent=2)+'\n')
    with (job/'matrix-controller.log').open('x') as log:
        subprocess.run(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=86400)


if __name__=='__main__':main()
