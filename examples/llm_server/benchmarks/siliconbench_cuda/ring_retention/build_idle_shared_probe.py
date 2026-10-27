"""Build a probe-only idle maintenance experiment without changing the SDK/PTE."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    original=args.parent/'mini_probe.cpp'
    assert sha(original)=='7992b817cabe7fb416a1f2cfd060b8cb9e0ae38994b09215b35f8782964e092b'
    before=original.read_text();after='#include "idle_shared_workset.h"\n'+before
    anchor='      } else if (op == "compact") {'
    assert after.count(anchor)==1
    body='''      } else if (op == "idle_refresh") {
        const auto refresh = siliconbench::refresh_idle_shared_workset();
        require(refresh.error == cudaSuccess, "idle shared pool refresh");
        result["idle_refresh"] = {{"enabled", refresh.enabled},
            {"used", refresh.used}, {"used_high", refresh.used_high},
            {"reserved_before", refresh.reserved_before},
            {"reserved_after", refresh.reserved_after}, {"keep", refresh.keep}};
'''
    after=after.replace(anchor,body+anchor)
    cpp=args.output/'idle_probe.cpp';cpp.write_text(after)
    header=Path(__file__).with_name('idle_shared_workset.h')
    shutil.copy2(header,args.output/header.name)
    flagsfile=args.parent/'probe-build/CMakeFiles/rewind_kv_probe.dir/flags.make'
    lines=flagsfile.read_text().splitlines();flags=[]
    for name in ('CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS'):
        flags+=shlex.split(next(x.split(' = ',1)[1] for x in lines if x.startswith(name+' = ')))
    obj=args.output/'idle_probe.cpp.o'
    commands=[['/usr/bin/c++',*flags,'-iquote',str(args.output),'-o',str(obj),'-c',str(cpp)]]
    cwd=args.parent/'probe-build'
    linkfile=cwd/'CMakeFiles/rewind_kv_probe.dir/link.txt'
    argv=[]
    for x in shlex.split(linkfile.read_text()):
        if x.startswith('-Wl,--dependency-file='):continue
        if x.endswith('.cpp.o'):x=str(obj)
        elif not x.startswith('-') and x.endswith(('.o','.a','.so')) and not Path(x).is_absolute():x=str((cwd/x).resolve())
        argv.append(x)
    probe=args.output/'rewind_kv_probe';argv[argv.index('-o')+1]=str(probe)
    assert str(obj) in argv;commands.append(argv)
    (args.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    with (args.output/'build.log').open('x') as log:
        for argv in commands:subprocess.run(argv,cwd=cwd,stdout=log,stderr=subprocess.STDOUT,check=True)
    gate=dict(probe=str(probe),probe_SHA256=sha(probe),source=str(args.parent/'native/executorch'),
        original_probe_SHA256=sha(original),candidate_probe_CPP_SHA256=sha(cpp),helper_SHA256=sha(header),
        original_flags_SHA256=sha(flagsfile),original_link_SHA256=sha(linkfile),
        SDK_changed=False,PTE_reexported=False,new_native_tests_claimed=False,
        qualification='Probe-only explicit idle maintenance feasibility; real-model gates pending')
    (args.output/'native-gate.json').write_text(json.dumps(gate,indent=2)+'\n')

if __name__=='__main__':main()
