"""Build pinned stock llama-server and collect one-round MG/Gemma matrices."""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

from bootstrap_current_native_chat import CPP_COMMIT,download_gguf,sha,write_wrapper


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','assets','lock'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--reuse-Gemma-gguf',type=Path)
    parser.add_argument('--port',type=int,required=True)
    args=parser.parse_args()
    assert not args.output.exists() and not args.assets.exists()
    args.output.mkdir(parents=True)
    args.assets.mkdir(parents=True)
    package=Path(__file__).resolve().parent
    def state(status,**extra):
        pending=args.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted=False,**extra),indent=2)+'\n')
        pending.replace(args.output/'progress.json')
    def run(label,command,timeout):
        (args.output/(label+'-command.json')).write_text(json.dumps(dict(argv=list(map(str,command))),indent=2)+'\n')
        with (args.output/(label+'.log')).open('x') as log:
            subprocess.run(list(map(str,command)),stdout=log,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,check=True,timeout=timeout)
    try:
        state('preparing-stock-CPP-and-public-GGUF')
        source=args.assets/'llama.cpp'
        build=args.assets/'build'
        def compile_server():
            run('clone-stock-CPP',['git','clone','--filter=blob:none','--no-checkout',
                'https://github.com/ggml-org/llama.cpp.git',source],900)
            run('checkout-stock-CPP',['git','-C',source,'checkout','--detach',CPP_COMMIT],600)
            agents=source/'AGENTS.md'
            if agents.is_file():
                (args.output/'stock-CPP-AGENTS.md').write_bytes(agents.read_bytes())
                print(agents.read_text(),flush=True)
            assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()==CPP_COMMIT
            run('configure-stock-CPP',['cmake','-S',source,'-B',build,'-DGGML_CUDA=ON',
                '-DGGML_CUDA_FORCE_CUBLAS=OFF','-DGGML_CUDA_FORCE_MMQ=OFF','-DGGML_CUDA_GRAPHS=ON',
                '-DCMAKE_CUDA_ARCHITECTURES=120','-DCMAKE_BUILD_TYPE=Release','-DLLAMA_CURL=OFF',
                '-DLLAMA_BUILD_TESTS=OFF'],900)
            run('build-stock-CPP',['cmake','--build',build,'--target','llama-server','--parallel','4'],7200)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(compile_server),pool.submit(download_gguf,'muse-glimmer',args.assets/'models'),
                     pool.submit(download_gguf,'gemma4',args.assets/'models',args.reuse_Gemma_gguf)]
            results=[f.result() for f in futures]
        binary=build/'bin/llama-server'
        assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True).strip()
        cmake=(build/'CMakeCache.txt').read_text()
        assert 'GGML_CUDA_FORCE_CUBLAS:BOOL=OFF' in cmake and 'GGML_CUDA:BOOL=ON' in cmake
        inputs=args.inputs or args.output/'inputs'
        if args.inputs is None:
            run('inputs',[sys.executable,'-B',package/'prepare_independent_main_public_inputs.py','--output',inputs],900)
        tasks=[]
        for model,alias,label,helper,client,prefill,proof in [
            ('muse-glimmer','muse-glimmer-30b','MG','prepare_mg_canonical_tokenizer.py','benchmark_mg.py',4096,results[1]),
            ('gemma4','gemma4-12b-it','Gemma','prepare_gemma_canonical_tokenizer.py','benchmark_gemma.py',2048,results[2])]:
            run(label+'-tokenizer',[sys.executable,'-B',package/helper,'--output',args.output/(label+'-tokenizer')],900)
            wrapper=args.output/(label+'-serve-stock.sh')
            write_wrapper(wrapper,model,'llamacpp',args.output/(label+'-tokenizer/tokenizer'),
                          Path(proof['path']),binary,args.output/(label+'-cache'))
            tasks.append(dict(name=label+'-llamacpp',model=alias,backend='llamacpp',
                wrapper=str(wrapper),client=str(package/client),inputs=str(inputs),
                benchmark_python=str(inputs/'.venv/bin/python'),prefill_budget=prefill,
                quality_policy='smoke',stock_framework_code_unmodified=True,GGUF=proof,
                CPP_commit=CPP_COMMIT,CPP_binary_SHA256=sha(binary),CUBLAS_forced=False))
        config=args.output/'matrix-config.json'
        config.write_text(json.dumps(dict(tasks=tasks,port=args.port,lock=str(args.lock)),indent=2)+'\n')
        state('running-single-round-matrix-before-deferred-quality',matrix_config=str(config))
        run('matrix',[sys.executable,'-B','-u',package/'run_default_chunk_single_matrix.py',
                      '--config',config,'--output',args.output/'matrix'],86400)
        state('stock-default-CPP-matrix-and-quality-collected')
    except BaseException as error:
        state('failed-preserve-existing-results',error=repr(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
