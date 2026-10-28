"""Build old and optional-idle real mini servers using immutable SDK archives."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('parent','overlay','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    tested=json.loads((args.overlay/'result.json').read_text())
    assert tested['cases']==6 and tested['fake_executor'] and tested['binary_SHA256']==sha(args.overlay/'server_idle_lifecycle_test')
    sdk=args.parent/'native/executorch';pkg=Path(__file__).parent
    for rel,meta in tested['source_manifest'].items():
        assert sha(sdk/rel)==meta['original_SHA256']
        assert sha(args.overlay/'include/executorch'/rel)==meta['candidate_SHA256']
    flagsfile=args.parent/'probe-build/CMakeFiles/rewind_kv_probe.dir/flags.make'
    lines=flagsfile.read_text().splitlines();flags=[]
    for n in ('CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS'):
        flags+=shlex.split(next(x.split(' = ',1)[1] for x in lines if x.startswith(n+' = ')))
    cwd=args.parent/'probe-build';linkfile=cwd/'CMakeFiles/rewind_kv_probe.dir/link.txt'
    commands=[];gate={}
    serving=sdk/'cmake-out/lib/libextension_llm_serving.a';assert serving.is_file()
    for arm in ('old','new'):
        obj=args.output/(arm+'.o');binary=args.output/(arm+'-server-mini')
        opt=['-DSB_SERVER_IDLE_HOOK='+('1' if arm=='new' else '0')]
        if arm=='new':opt.insert(0,'-I'+str(args.overlay/'include'))
        commands.append(['/usr/bin/c++',*opt,*flags,'-iquote',str(pkg),'-o',str(obj),'-c',str(pkg/'server_idle_real_mini.cpp')])
        argv=[]
        for x in shlex.split(linkfile.read_text()):
            if x.startswith('-Wl,--dependency-file='):continue
            if x.endswith('.cpp.o'):
                argv.append(str(obj))
                if arm=='new':argv.extend([str(args.overlay/'runner.o'),str(args.overlay/'serving_runtime.o')])
                argv.append(str(serving));continue
            if not x.startswith('-') and x.endswith(('.a','.o','.so')) and not Path(x).is_absolute():x=str((cwd/x).resolve())
            argv.append(x)
        argv[argv.index('-o')+1]=str(binary);commands.append(argv)
        gate[arm]=dict(binary=str(binary))
    (args.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    with (args.output/'build.log').open('x') as log:
        for argv in commands:subprocess.run(argv,cwd=cwd,check=True,stdout=log,stderr=subprocess.STDOUT)
    for meta in gate.values():meta['SHA256']=sha(meta['binary'])
    gate.update(source_CPP_SHA256=sha(pkg/'server_idle_real_mini.cpp'),idle_helper_SHA256=sha(pkg/'idle_shared_workset.h'),quiet_policy_SHA256=sha(pkg/'idle_shared_quiet_policy.h'),
                overlay_tests_SHA256=sha(args.overlay/'result.json'),overlay_objects={n:sha(args.overlay/n) for n in ('runner.o','serving_runtime.o')},
                original_SDK_modified=False,PTE_reexported=False)
    (args.output/'native-gate.json').write_text(json.dumps(gate,indent=2)+'\n');print(json.dumps(gate,indent=2))

if __name__=='__main__':main()
