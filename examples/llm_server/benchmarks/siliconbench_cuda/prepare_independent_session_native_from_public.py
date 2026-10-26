"""Prepare the public ABI2 base in an isolated copy and launch ABI3 native tests.

Reuses checked-out public dependencies. Full-model export is unavailable.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-source', type=Path, required=True)
    parser.add_argument('--python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lock', type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    spec = json.loads((package/'runtime_kv_columns_source.json').read_text())
    assert not args.output.exists() and args.baseline_source.is_dir()
    assert digest(package/'runtime_kv_columns.patch') == spec['patch_sha256']
    args.output.mkdir(parents=True)
    base = args.output/'ABI2-base'

    def ignore(directory,names):
        return [name for name in names if name in
                ['.git','cmake-out','__pycache__','.pytest_cache'] or
                name.endswith(('.pte','.ptd','.gguf','.log','.nsys-rep'))]

    shutil.copytree(args.baseline_source,base,symlinks=True,ignore=ignore)
    if any(not (base/'third-party'/name/'CMakeLists.txt').is_file()
           for name in ['googletest','flatbuffers']):
        subprocess.run(['git','init','-q',str(base)],check=True)
        subprocess.run(['git','-C',str(base),'remote','add','origin',
                        'https://github.com/pytorch/executorch.git'],check=True)
        subprocess.run(['git','-C',str(base),'fetch','--filter=blob:none','--depth=1',
                        'origin',spec['base_commit']],check=True,timeout=600)
        subprocess.run(['git','-C',str(base),'reset','--mixed',spec['base_commit']],
                       check=True,stdout=subprocess.DEVNULL)
        modules = subprocess.check_output(['git','-C',str(base),'show',
                    spec['base_commit']+':.gitmodules'])
        (base/'.gitmodules').write_bytes(modules)
        urls = subprocess.check_output(['git','config','--file',str(base/'.gitmodules'),
                '--get-regexp','^submodule[.].*[.]url$'],text=True)
        assert all(line.split(maxsplit=1)[1].startswith('https://github.com/')
                   for line in urls.splitlines()), urls
        subprocess.run(['git','-C',str(base),'submodule','update','--init',
                        '--recursive','--depth=1','--jobs=4'],check=True,timeout=3600)
    original = {}
    for name,record in spec['changed_files'].items():
        p = base/name
        expected = record['before_sha256']
        if expected is None:
            if p.exists():
                assert digest(p)==record['after_sha256'],name
                p.unlink()
            continue
        original[name] = digest(args.baseline_source/name)
        raw = subprocess.check_output(['git','-C',str(args.baseline_source),
                                       'show',spec['base_commit']+':'+name])
        assert hashlib.sha256(raw).hexdigest()==expected,name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(raw)
    for suffix in ['program.fbs','scalar_type.fbs']:
        resource=base/'exir/_serialize'/suffix
        if not resource.exists():
            shutil.copy2(base/'schema'/suffix,resource)
    alias=base/'examples/models/muse_glimmer'
    if not alias.exists():
        assert not alias.is_symlink()
        alias.symlink_to('muse-glimmer')
    subprocess.run(['git','apply','--check',str(package/'runtime_kv_columns.patch')],
                   cwd=base,check=True)
    subprocess.run(['git','apply',str(package/'runtime_kv_columns.patch')],
                   cwd=base,check=True)
    for name,record in spec['changed_files'].items():
        assert digest(base/name)==record['after_sha256'],name
    for name,sha in original.items():
        assert digest(args.baseline_source/name)==sha,name
    argv=[str(args.python),'-B','-u',str(package/'prepare_independent_session_mini.py'),
          '--baseline-source',str(base),'--python',str(args.python),
          '--output',str(args.output/'native-build'),'--lock',str(args.lock),'--jobs','4']
    with (args.output/'native-controller.log').open('x') as log:
        child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,
                               stderr=subprocess.STDOUT,start_new_session=True)
    receipt=dict(native_controller_pid=child.pid,argv=argv,
                 public_base_commit=spec['base_commit'],
                 preserved_baseline_hashes=original,baseline_modified=False,
                 full_model_export_started=False)
    (args.output/'deployment.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':
    main()
