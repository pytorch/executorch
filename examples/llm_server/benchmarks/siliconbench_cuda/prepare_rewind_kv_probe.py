"""Add a rewind protocol operation and optional sampler to an isolated probe.

Use the unchanged qualified native SDK. Do not edit original source or binaries,
alter exported methods, or replace the server. This is diagnostic-only tooling.
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
    h=hashlib.sha256()
    with Path(path).open('rb') as file:
        while block:=file.read(16*1024*1024):h.update(block)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('preparation-job','python','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    package=Path(__file__).resolve().parent
    def state(status,**fields):
        pending=args.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),CPU_only=True,
            candidate_accepted=False,**fields),indent=2)+'\n')
        pending.replace(args.output/'progress.json')
    try:
        state('verifying-unchanged-native-SDK')
        gate_path=args.preparation_job/'native-gate.json'
        gate=json.loads(gate_path.read_text())
        assert gate['native_GPU_tests_pass'] and gate['test_count']==74
        source=Path(gate['source'])
        sdk=source/'cmake-out'
        libraries={str(path):digest(path) for path in (sdk/'lib').glob('*.a')}
        assert libraries
        for name,expected in gate['binaries'].items():assert digest(name)==expected,name
        for name,expected in gate['unchanged_controls'].items():assert digest(source/name)==expected,name
        official=source/'examples/models/muse-glimmer/tests/runtime_kv_columns_probe.cpp'
        spec=json.loads((package/'runtime_kv_columns_source.json').read_text())
        official_expected=spec['changed_files']['examples/models/muse-glimmer/tests/runtime_kv_columns_probe.cpp']['after_sha256']
        assert digest(official)==official_expected
        original=official.read_text()
        modified=original
        needle='#include <unistd.h>'
        assert modified.count(needle)==1
        modified=modified.replace(needle,'#include "kv_memory_sampler.h"\n'+needle)
        needle='    require(executor->initialize(), "initialize actual CUDA executor");'
        assert modified.count(needle)==1
        modified=modified.replace(needle,needle+'\n    siliconbench::KVMemorySampler sampler(*executor, "cuda", false);')
        needle='      } else if (op == "compact") {'
        assert modified.count(needle)==1
        modified=modified.replace(needle,
            '      } else if (op == "rewind") {\n'
            '        const auto sid = request.at("seq_id").get<int32_t>();\n'
            '        const bool rewound = control.rewind(sid, request.at("position").get<int>());\n'
            '        require(rewound == request.value("expect_success", true), "rewind expectation");\n'
            '        result["rewound"] = rewound;\n'
            '        result["position"] = control.pos(sid);\n'+needle)
        cpp=args.output/'rewind_kv_probe.cpp'
        cpp.write_text(modified)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','CUDA_HOME':'/usr/local/cuda',
             'CUDAARCHS':'120','PYTHONDONTWRITEBYTECODE':'1'}
        env['PATH']=str(args.python.parent)+os.pathsep+env['PATH']
        cmake=shutil.which('cmake',path=env['PATH'])
        assert cmake is not None
        prefix=subprocess.check_output([str(args.python),'-B','-c',
            'import torch;print(torch.utils.cmake_prefix_path)'],env=env,text=True).strip()
        build=args.output/'build'
        commands=[[cmake,'-S',str(package/'rewind_kv_probe_project'),'-B',str(build),
            '-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON',
            '-DET_SOURCE='+str(source),'-DREWIND_PROBE_SOURCE='+str(cpp),
            '-DCMAKE_PREFIX_PATH='+str(sdk)+';'+prefix],
            [cmake,'--build',str(build),'--parallel','2','--target','rewind_kv_probe']]
        state('building-isolated-rewind-probe')
        with (args.output/'build.log').open('x') as log:
            for command in commands:
                log.write(json.dumps(command)+'\n');log.flush()
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,
                               check=True,timeout=3600)
        for name,expected in libraries.items():assert digest(name)==expected,name
        for name,expected in gate['binaries'].items():assert digest(name)==expected,name
        for name,expected in gate['unchanged_controls'].items():assert digest(source/name)==expected,name
        assert digest(official)==official_expected
        binary=build/'rewind_kv_probe'
        original_probe=args.preparation_job/'model-build/muse_glimmer_runtime_kv_probe'
        proof=dict(native_gate_SHA256=digest(gate_path),source=str(source),
            unchanged_SDK_libraries=libraries,original_probe_SHA256=digest(original_probe),
            probe=str(binary),probe_SHA256=digest(binary),generated_CPP_SHA256=digest(cpp),
            official_probe_CPP_SHA256=digest(official),
            optional_sampler_header_SHA256=digest(package/'kv_memory_sampler.h'),
            only_protocol_rewind_and_optional_sampler_added=True,no_inference_kernel_change=True,
            no_original_source_or_binary_changed=True,CUDA_hidden_during_build=True,
            model_validation_pending=True,compile_commands_SHA256=digest(build/'compile_commands.json'))
        (args.output/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
        state('isolated-rewind-probe-ready',probe=str(binary))
    except Exception as error:
        state('failed',detail=str(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
