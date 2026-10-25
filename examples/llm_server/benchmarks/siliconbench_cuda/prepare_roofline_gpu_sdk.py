"""Build a separate ET API SDK and stock default CUDA llama.cpp for roofline.

Wait for the isolated Python environment before building. CUDA devices stay
hidden. Two independent CPU builds can overlap operator screening; no driver,
framework source or existing model directory is changed.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['root','et-source','python','setup-progress','output']:p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--cpp-commit',default='620d2225cc436ab0b001e3200e590e3b286ef9ab')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    assert len(a.cpp_commit)==40 and all(c in '0123456789abcdef' for c in a.cpp_commit)
    status=dict(controller_pid=os.getpid(),phases={},CPU_only=True,CPP_FORCE_CUBLAS=False,main_results_changed=False)
    guard=threading.Lock()
    def state(phase,value,**fields):
        with guard:
            status['phases'][phase]=value;status.update(fields,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
            tmp=a.output/'progress.pending.json';tmp.write_text(json.dumps(status,indent=2)+'\n');tmp.replace(a.output/'progress.json')
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUDAARCHS='120',TORCH_CUDA_ARCH_LIST='12.0',CUDA_HOME='/usr/local/cuda',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MAX_JOBS='4')
    env['PATH']=str(a.python.parent)+os.pathsep+env['PATH']
    def run(command,logname,timeout=7200):
        with (a.output/logname).open('a') as log:
            log.write(json.dumps([str(v) for v in command])+'\n');log.flush()
            subprocess.run([str(v) for v in command],env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=timeout)
    def ET():
        state('ET','waiting-isolated-Python-environment')
        start=time.monotonic()
        while True:
            progress=json.loads(a.setup_progress.read_text())
            assert progress['phases'].get('controller')!='failed','Inspect preserved setup attempt'
            if progress['phases'].get('environment')=='ready':break
            assert time.monotonic()-start<3600;time.sleep(20)
        prefix=subprocess.check_output([str(a.python),'-B','-c','import torch;print(torch.utils.cmake_prefix_path)'],env=env,text=True).strip()
        build=a.output/'ET-core';sdk=a.output/'ET-sdk'
        state('ET','building-isolated-native-SDK')
        run(['cmake','-S',a.et_source,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=120','-DEXECUTORCH_BUILD_PRESET_FILE='+str(a.et_source/'tools/cmake/preset/llm.cmake'),'-DEXECUTORCH_BUILD_CUDA=ON','-DEXECUTORCH_BUILD_PYBIND=OFF','-DEXECUTORCH_BUILD_TESTS=OFF','-DBUILD_TESTING=OFF','-DCMAKE_INSTALL_PREFIX='+str(sdk),'-DCMAKE_PREFIX_PATH='+prefix,'-DPYTHON_EXECUTABLE='+str(a.python)],'ET-build.log')
        run(['cmake','--build',build,'--parallel','4','--target','install'],'ET-build.log')
        project=Path(__file__).resolve().parent/'roofline_sdk_project';api=a.output/'ET-API'
        state('ET','building-official-API-driver')
        run(['cmake','-S',project,'-B',api,'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DET_SOURCE='+str(a.et_source),'-DCMAKE_PREFIX_PATH='+str(sdk)+';'+prefix],'ET-build.log')
        run(['cmake','--build',api,'--parallel','4','--target','roofline_et'],'ET-build.log')
        binaries={str(api/'roofline_et'):sha(api/'roofline_et')}
        (a.output/'ET-provenance.json').write_text(json.dumps(dict(binary=binaries,SDK=str(sdk),source_HEAD=subprocess.check_output(['git','-C',str(a.et_source),'rev-parse','HEAD'],text=True).strip(),CMake_cache_SHA256=sha(build/'CMakeCache.txt'),compile_commands_SHA256=sha(api/'compile_commands.json')),indent=2)+'\n')
        state('ET','ready',ET_binary=str(api/'roofline_et'))
    def CPP():
        state('CPP','building-stock-CUDA-default')
        source=a.output/'llama.cpp';build=a.output/'CPP-build'
        run(['git','clone','--filter=blob:none','--no-checkout','https://github.com/ggml-org/llama.cpp.git',source],'CPP-build.log')
        run(['git','-C',source,'fetch','--depth','1','origin',a.cpp_commit],'CPP-build.log')
        run(['git','-C',source,'checkout','--detach',a.cpp_commit],'CPP-build.log')
        run(['cmake','-S',source,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=120','-DGGML_CUDA=ON','-DGGML_CUDA_FORCE_CUBLAS=OFF','-DGGML_NATIVE=ON','-DBUILD_SHARED_LIBS=ON','-DLLAMA_BUILD_TESTS=OFF'],'CPP-build.log')
        run(['cmake','--build',build,'--parallel','4','--target','llama'],'CPP-build.log')
        cache=(build/'CMakeCache.txt').read_text();assert 'GGML_CUDA:BOOL=ON' in cache and 'GGML_CUDA_FORCE_CUBLAS:BOOL=OFF' in cache
        package=Path(__file__).resolve().parent;binary=a.output/'roofline_cpp_default'
        lib=build/'bin'
        run(['/usr/bin/c++','-O3','-DNDEBUG','-std=c++20','-pthread','-I'+str(source/'include'),'-I'+str(source/'ggml/include'),'-I/usr/local/cuda/include',package/'roofline_active_cpp.cpp','-L'+str(lib),'-Wl,-rpath,'+str(lib),'-lllama','-L/usr/local/cuda/lib64','-Wl,-rpath,/usr/local/cuda/lib64','-lcudart','-o',binary],'CPP-build.log')
        assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip()
        (a.output/'CPP-provenance.json').write_text(json.dumps(dict(binary_SHA256=sha(binary),commit=a.cpp_commit,CMake_cache_SHA256=sha(build/'CMakeCache.txt'),CUDA=True,FORCE_CUBLAS=False,source_unmodified=True),indent=2)+'\n')
        state('CPP','ready',CPP_binary=str(binary))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(ET),pool.submit(CPP)]
            for future in futures:future.result()
        state('controller','completed-pending-GPU-runs')
    except BaseException as error:
        state('controller','failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
