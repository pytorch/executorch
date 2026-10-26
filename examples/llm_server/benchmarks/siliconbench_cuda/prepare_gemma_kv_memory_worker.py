"""Build the existing Gemma multiplexed worker with optional KV sampling.

Use the qualified unchanged native SDK and the existing Gemma llm_server
adapter. Add only a lifetime-bound sampler disabled unless explicitly enabled.
Preserve tokenizer, scheduler, stops, transport, original sources and SDK.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as file:
        while block:=file.read(16*1024*1024):
            value.update(block)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('native-job','python','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    package=Path(__file__).resolve().parent
    def state(status,**fields):
        pending=args.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_only=True,
            candidate_accepted=False,full_model_export_started=False,**fields),indent=2)+'\n')
        pending.replace(args.output/'progress.json')
    try:
        state('verifying-qualified-Gemma-native-SDK')
        gate_path=args.native_job/'native-gate.json'
        gate=json.loads(gate_path.read_text())
        assert gate['native_GPU_tests_pass'] and gate['test_count']==74
        progress=json.loads((args.native_job/'progress.json').read_text())
        assert progress['status']=='native-GPU-tests-passed'
        source=Path(gate['source'])
        sdk=source/'cmake-out'
        libraries={str(path):digest(path) for path in (sdk/'lib').glob('*.a')}
        assert libraries
        for name,expected in gate['binaries'].items():assert digest(name)==expected,name
        for name,expected in gate['unchanged_controls'].items():assert digest(source/name)==expected,name
        public=json.loads((package/'public-source-manifest.json').read_text())
        original=package/'gemma_runtime_columns_worker.cpp'
        assert digest(original)==public['files'][original.name]['sha256']
        modified=original.read_text()
        include='#include <unistd.h>'
        initialize='    llm::serving::ServingRuntime runtime(*executor, std::move(scheduler), tokenizer, config);'
        assert modified.count(include)==1 and modified.count(initialize)==1
        modified=modified.replace(include,'#include "kv_memory_sampler.h"\n'+include)
        modified=modified.replace(initialize,initialize+
            '\n    siliconbench::KVMemorySampler memory_sampler(*executor, "cuda", false);')
        cpp=args.output/'gemma_kv_memory_worker.cpp'
        cpp.write_text(modified)
        sampler=(package/'kv_memory_sampler.h').read_text()
        assert 'cudaDeviceSynchronize' not in sampler and 'cudaStreamSynchronize' not in sampler
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','CUDA_HOME':'/usr/local/cuda',
             'CUDAARCHS':'120','PYTHONDONTWRITEBYTECODE':'1'}
        env['PATH']=str(args.python.parent)+os.pathsep+env['PATH']
        cmake=shutil.which('cmake',path=env['PATH'])
        assert cmake is not None
        prefix=subprocess.check_output([str(args.python),'-B','-c',
            'import torch;print(torch.utils.cmake_prefix_path)'],env=env,text=True).strip()
        build=args.output/'build'
        commands=[[cmake,'-S',str(package/'gemma_runtime_worker_project'),'-B',str(build),
            '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
            '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
            '-DET_SOURCE='+str(source),'-DGEMMA_WORKER_SOURCE='+str(cpp),
            '-DCMAKE_PREFIX_PATH='+str(sdk)+';'+prefix],
            [cmake,'--build',str(build),'--parallel','2','--target','gemma_runtime_columns_worker']]
        state('building-isolated-Gemma-memory-worker')
        with (args.output/'build.log').open('x') as log:
            for command in commands:
                log.write(json.dumps(command)+'\n');log.flush()
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                               check=True,timeout=3600)
        for name,expected in libraries.items():assert digest(name)==expected,name
        for name,expected in gate['binaries'].items():assert digest(name)==expected,name
        for name,expected in gate['unchanged_controls'].items():assert digest(source/name)==expected,name
        assert digest(original)==public['files'][original.name]['sha256']
        binary=build/'gemma_runtime_columns_worker'
        proof=dict(native_gate_SHA256=digest(gate_path),source=str(source),SDK=str(sdk),
            unchanged_SDK_libraries=libraries,instrumented_worker=str(binary),
            instrumented_worker_SHA256=digest(binary),generated_CPP_SHA256=digest(cpp),
            original_Gemma_worker_source_SHA256=digest(original),
            optional_sampler_header_SHA256=digest(package/'kv_memory_sampler.h'),
            existing_Gemma_llm_server_adapter_preserved=True,
            scheduler_stops_tokenizer_transport_unchanged=True,
            no_inference_kernel_change=True,sampler_does_not_synchronize_GPU=True,
            optional_native_sampler_defaults_off=True,sample_interval_ms=250,
            original_source_and_SDK_unchanged=True,CUDA_hidden_during_build=True,
            HTTP_and_sampling_performance_validation_pending=True,
            compile_commands_SHA256=digest(build/'compile_commands.json'))
        (args.output/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
        state('Gemma-memory-worker-ready-pending-HTTP-validation',worker=str(binary))
    except Exception as error:
        state('failed',detail=str(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
