"""Compile only an isolated idle overlay against the existing qualified SDK."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    sdk=args.parent/'native/executorch';sources=Path(__file__).parent
    overlay=args.output/'include/executorch'
    subprocess.run(['python3','-B',str(sources/'apply_server_idle_patch.py'),'--sdk',str(sdk),'--output',str(overlay)],check=True)
    manifest=json.loads((overlay/'manifest.json').read_text())
    expected={'extension/llm/batching/runner.h':'32ec2493ac28abb31c3f789f35798d6912dbcc046ec17384f782cc4b3295a484',
              'extension/llm/batching/runner.cpp':'2a0314b227ba7633fe486e10518a76cffe9ced7effeb35e163b1470b2a61262e',
              'extension/llm/serving/serving_runtime.h':'91ee3da837addd57da8627cd3965b015199e7dbfd316f064948ad963e19cff99',
              'extension/llm/serving/serving_runtime.cpp':'3b8a3b1ef5f7ed82783be5a34dd8e4eb223c07325ed78f0ad704228c69213199'}
    assert {r:m['original_SHA256'] for r,m in manifest.items()}==expected
    flagsfile=args.parent/'probe-build/CMakeFiles/rewind_kv_probe.dir/flags.make'
    lines=flagsfile.read_text().splitlines();flags=[]
    for n in ('CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS'):
        flags+=shlex.split(next(x.split(' = ',1)[1] for x in lines if x.startswith(n+' = ')))
    flags=['-I'+str(args.output/'include'),*flags]
    objects=[];commands=[]
    for cpp in [overlay/'extension/llm/batching/runner.cpp',overlay/'extension/llm/serving/serving_runtime.cpp',sources/'server_idle_lifecycle_test.cpp']:
        obj=args.output/(cpp.stem+'.o');objects.append(obj)
        commands.append(['/usr/bin/c++',*flags,'-o',str(obj),'-c',str(cpp)])
    linkfile=args.parent/'probe-build/CMakeFiles/rewind_kv_probe.dir/link.txt'
    cwd=args.parent/'probe-build';argv=[]
    serving=sdk/'cmake-out/lib/libextension_llm_serving.a'
    assert serving.is_file(), 'Read current link/build artifacts to locate serving archive'
    for x in shlex.split(linkfile.read_text()):
        if x.startswith('-Wl,--dependency-file='):continue
        if x.endswith('.cpp.o'):
            argv.extend([str(o) for o in objects]+[str(serving)]);continue
        if not x.startswith('-') and x.endswith(('.a','.o','.so')) and not Path(x).is_absolute():x=str((cwd/x).resolve())
        argv.append(x)
    binary=args.output/'server_idle_lifecycle_test';argv[argv.index('-o')+1]=str(binary);commands.append(argv)
    (args.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    with (args.output/'build.log').open('x') as log:
        for argv in commands:subprocess.run(argv,cwd=cwd,check=True,stdout=log,stderr=subprocess.STDOUT)
    r=subprocess.run([str(binary)],text=True,capture_output=True,timeout=30)
    (args.output/'test.stdout').write_text(r.stdout);(args.output/'test.stderr').write_text(r.stderr)
    assert r.returncode==0,r.stderr
    result=json.loads(r.stdout.splitlines()[-1]);assert result['cases']==6 and result['fake_executor'] and not result['GPU_model_tested']
    result.update(binary_SHA256=sha(binary),source_manifest=manifest,SDK_changed=False,PTE_reexported=False,
                  new_GPU_performance_claimed=False,commands_SHA256=sha(args.output/'commands.json'))
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
