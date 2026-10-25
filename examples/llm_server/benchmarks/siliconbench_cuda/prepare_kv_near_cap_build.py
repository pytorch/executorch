"""Compile an isolated near-cap row policy and probes using existing inputs.

Only the owned archive's cuda_kv_pool object changes. Existing headers, worker
object, libraries, environments, model artifacts, and source stay untouched.
This CPU preparation must be serialized with performance work by its caller.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['et-source','build-inputs','checkpoint-build','output']:
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--recorded-et-source',required=True)
    a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    et=a.et_source.resolve();inputs=a.build_inputs.resolve()
    archive=inputs/'lib/libaoti_cuda_backend.a'
    original=et/'backends/cuda/runtime/cuda_kv_pool.cpp'
    assert digest(original)=='45f9c2937dad2ba75fb65014c740e9538448bcb6b4e00cc27918b8ada3f627d6'
    assert subprocess.check_output(['ar','t',str(archive)],text=True).splitlines().count('cuda_kv_pool.cpp.o')==1
    worker_obj=a.checkpoint_build/'worker.o'
    assert worker_obj.is_file() and (a.checkpoint_build/'gemma4_checkpoint_worker').is_file()
    watched=[original,et/'backends/cuda/runtime/cuda_kv_pool.h',worker_obj,inputs/'build-audit.json',
             *[x for x in (inputs/'lib').iterdir() if x.is_file()]]
    before={str(x):digest(x) for x in watched}
    src=original.read_text();marker='    const int64_t rows = std::min(new_rows, layer.declared_rows);'
    assert src.count(marker)==1
    patched=src.replace(marker,'''    int64_t rows = std::min(new_rows, layer.declared_rows);
    // Avoid a separate near-identical whole-layer allocation at the next
    // geometric growth. Initial allocation and logical row capacity do not
    // change. Extra physical rows are at most1/32 of this growth's target.
    if (layer.growable && rows > allocations_.at(index).rows &&
        layer.declared_rows - rows > 0 &&
        layer.declared_rows - rows <= rows / 32) {
      rows = layer.declared_rows;
    }''')
    audit=json.loads((inputs/'build-audit.json').read_text())
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    commands=[]
    def run(argv):
        commands.append(argv)
        with (a.output/'build.log').open('a') as log:
            log.write(json.dumps(argv)+'\n');log.flush()
            subprocess.run(argv,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    def compile(source,obj):
        cmd=audit['compile_command'].copy();recorded=a.recorded_et_source
        for i,x in enumerate(cmd):
            cmd[i]=x.replace(recorded+'/cmake-out/include',str(inputs/'include')).replace(
                recorded+'/cmake-out/third-party/gflags/include',str(inputs/'gflags-include')).replace(recorded,str(et))
        cmd[cmd.index('-c')+1]=str(source);cmd[cmd.index('-o')+1]=str(obj);run(cmd)
    def link(obj,binary,lib):
        cmd=audit['link_command'].copy();recorded=a.recorded_et_source
        for i,x in enumerate(cmd):
            if x.endswith('.o'):cmd[i]=str(obj)
            elif x.startswith('lib') and x.endswith('.a'):cmd[i]=str(inputs/'lib'/x)
            elif recorded+'/cmake-out/lib/' in x or recorded+'/cmake-out/third-party/gflags/libgflags_' in x:
                start=x.index(recorded);end=x.find(',',start);end=len(x) if end<0 else end
                cmd[i]=x[:start]+str(inputs/'lib'/Path(x[start:end]).name)+x[end:]
            elif x.startswith('-Wl,-rpath,'):cmd[i]='-Wl,-rpath,'+str(inputs/'lib')+':/usr/local/cuda/lib64'
            elif x.startswith('/usr/local/cuda-13.2/') and not Path(x).exists():cmd[i]=x.replace('/usr/local/cuda-13.2/','/usr/local/cuda/')
            if cmd[i]==str(archive):cmd[i]=str(lib)
        cmd[cmd.index('-o')+1]=str(binary);run(cmd)
    probe_source=Path(__file__).with_name('kv_near_cap_probe.cpp')
    probe_obj=a.output/'probe.o';compile(probe_source,probe_obj)
    binaries={};archives={}
    for arm,text in [('baseline',src),('candidate',patched)]:
        d=a.output/arm;d.mkdir();source=d/'cuda_kv_pool.cpp';source.write_text(text)
        obj=d/'cuda_kv_pool.cpp.o';compile(source,obj)
        lib=d/'libaoti_cuda_backend.a';shutil.copy2(archive,lib);run(['ar','r',str(lib),str(obj)])
        assert subprocess.check_output(['ar','t',str(lib)],text=True).splitlines().count(obj.name)==1
        for member in subprocess.check_output(['ar','t',str(archive)],text=True).splitlines():
            if member != obj.name:
                assert subprocess.check_output(['ar','p',str(archive),member])==subprocess.check_output(['ar','p',str(lib),member])
        probe=d/'kv_near_cap_probe';link(probe_obj,probe,lib)
        binaries[arm]=dict(probe=str(probe),probe_SHA256=digest(probe),source_SHA256=digest(source))
        archives[arm]=dict(path=str(lib),SHA256=digest(lib))
        if arm=='candidate':
            worker=d/'gemma4_checkpoint_worker';link(worker_obj,worker,lib)
            binaries[arm].update(worker=str(worker),worker_SHA256=digest(worker))
    assert before=={str(x):digest(x) for x in watched}
    result=dict(status='CPU-build-completed',binaries=binaries,archives=archives,commands=commands,
        unchanged_inputs=before,original_sources_libraries_worker_unchanged=True,
        initial_allocation_unchanged=True,logical_row_count_unchanged=True,
        failure_rollback_semantics_unchanged=True,extra_rows_bound='at most target_rows/32',
        GPU_work=False,candidate_accepted=False)
    (a.output/'build-audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
