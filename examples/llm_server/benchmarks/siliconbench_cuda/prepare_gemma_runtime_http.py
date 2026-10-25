"""Prepare model Python and the official HTTP worker over a qualified KV core.

Only a terminal isolated source is updated. Preserve replaced Python files;
the original main checkout, CUDA libraries and model artifacts stay separate.
This CPU preparation does not qualify model quality or measured memory.
"""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['qualified-core','python','output']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--reuse-prepared-Python',action='store_true',
                        help='Verify already staged exact payloads without writing them again')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    package=Path(__file__).resolve().parent
    progress=dict(controller_pid=os.getpid(),CPU_only=True,HTTP_accepted=False)
    def state(status,**fields):
        progress.update(status=status,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**fields)
        (args.output/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
    try:
        core=json.loads(args.qualified_core.read_text());source=Path(core['source'])
        assert core['test_count']==25 and core['model_HTTP_qualified'] is False
        assert all(sha(source/name)==digest for name,digest in core['source_files'].items())
        tests=json.loads((args.qualified_core.parent/'GPU-tests.json').read_text())
        assert tests['failures']==0 and sha(args.qualified_core.parent/'GPU-tests.json')==core['GPU_test_report_SHA256']
        base=source/'examples/models/gemma4_31b'
        items={'batching_model.py':(None,'a3f1ab403eb9e49f1f74ca7842f7f1aad10808ed26ee21664091ee36ff2f1e81'),
               'batching_gguf_loader.py':(None,'4ae63dbb5f7ae2dbf900ea6b6da85d9c03ef2e28f814fb3c07d3231f3b33fcbc'),
               'cuda_source_transformations.py':('cee7e9e757146eb4e2356f320db8c0f00884f3574ddc973c4c369ce4c2ce52a0','5a3d60a6c6b8c645df54b006c1eda9ac83e880ff7f9de7c489c755a22e7f081c')}
        state('preserving-terminal-Python-source',source=str(source))
        old={}
        for name,(before,after) in items.items():
            original=base/name;payload=package/'gemma_runtime_python'/name
            assert sha(payload)==after;ast.parse(payload.read_text())
            if args.reuse_prepared_Python:
                assert sha(original)==after,name
                old[name]='already-staged-exact-payload'
                continue
            if before is None:assert not original.exists(),name
            else:
                assert sha(original)==before,name
                dest=args.output/'before-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,dest)
            old[name]=before;shutil.copy2(payload,original);assert sha(original)==after
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MAX_JOBS='2',PYTHONPATH=str(source.parent))
        prefix=subprocess.check_output([str(args.python),'-B','-c','import torch;print(torch.utils.cmake_prefix_path)'],env=env,text=True).strip()
        command=[str(args.python),'-B','-c','from executorch.examples.models.gemma4_31b import export_batching,serve;from executorch.examples.llm_server.python.server import build_app;print("Python and HTTP imports ready")']
        probe=subprocess.run(command,env=env,capture_output=True,text=True);(args.output/'import.log').write_text(probe.stdout+probe.stderr);assert probe.returncode==0
        build=args.output/'worker-build';sdk=source/'cmake-out'
        state('building-official-HTTP-worker')
        commands=[['cmake','-S',str(package/'gemma_runtime_worker_project'),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DET_SOURCE='+str(source),'-DCMAKE_PREFIX_PATH='+str(sdk)+';'+prefix],['cmake','--build',str(build),'--parallel','2','--target','gemma_runtime_columns_worker']]
        with (args.output/'build.log').open('a') as log:
            for cmd in commands:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1800)
        binary=build/'gemma_runtime_columns_worker'
        result=subprocess.run([str(binary),'--helpshort'],env=env,capture_output=True,text=True,timeout=30)
        assert 'idle_kv_compaction_ms' in result.stdout+result.stderr
        assert all(sha(source/name)==digest for name,digest in core['source_files'].items())
        receipt=dict(source=str(source),qualified_core_SHA256=sha(args.qualified_core),before_Python=old,after_Python={name:sha(base/name) for name in items},binary=str(binary),binary_SHA256=sha(binary),compile_commands_SHA256=sha(build/'compile_commands.json'),HTTP_accepted=False)
        (args.output/'build-provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
        state('CPU-HTTP-worker-ready-pending-model',binary=str(binary))
    except BaseException as error:
        state('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
