"""Prepare pinned stock MG/Gemma runtimes and run the default-budget matrix.

New outputs and caches are isolated. Existing model files and environments may
be reused read only. Performance precedes destination correctness checks.
"""
import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from bootstrap_corrected_native_MG_chat import download_model,sha
from bootstrap_current_native_chat import download_gemma,write_wrapper


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['vllm','sglang'],required=True)
    for name in ('output','native-venv','MG-model','Gemma-model','lock'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--inputs',type=Path)
    parser.add_argument('--after-performance',type=Path)
    parser.add_argument('--reuse-native',action='store_true')
    parser.add_argument('--reuse-MG-model',action='store_true')
    parser.add_argument('--port',type=int,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    args.output.mkdir(parents=True)
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
        state('preparing-stock-environment-and-public-models')
        uv=shutil.which('uv')
        assert uv and Path('/usr/local/cuda/bin/ptxas').is_file()
        native=args.native_venv/'bin/python'
        def install():
            torch='2.13.0' if args.backend=='vllm' else '2.14.1'
            runtime='0.32.0' if args.backend=='vllm' else '0.5.22'
            if not args.reuse_native:
                assert not args.native_venv.exists()
                run('create-native-venv',[uv,'venv','--python','3.12',args.native_venv],600)
                run('install-official-native',[uv,'pip','install','--python',native,
                    args.backend+'=='+runtime,'torch=='+torch,'transformers==5.19.0'],3600)
            run('verify-native',[native,'-B','-c',
                "import importlib.metadata as m,json,torch;v={n:m.version(n) for n in "
                +repr(['torch','transformers',args.backend])+"};print(json.dumps({'versions':v,'CUDA':torch.version.cuda}));assert v=="
                +repr({'torch':torch,'transformers':'5.19.0',args.backend:runtime})],180)
            run('package-freeze',[uv,'pip','freeze','--python',native],120)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures=[pool.submit(install),pool.submit(download_model,args.MG_model,args.reuse_MG_model),
                     pool.submit(download_gemma,args.Gemma_model)]
            for future in futures:future.result()
        state('preparing-pinned-inputs-and-tokenizers')
        inputs=args.inputs or args.output/'inputs'
        if args.inputs is None:
            run('inputs',[sys.executable,'-B',package/'prepare_independent_main_public_inputs.py','--output',inputs],900)
        for model,helper in [('MG','prepare_mg_canonical_tokenizer.py'),('Gemma','prepare_gemma_canonical_tokenizer.py')]:
            run(model+'-tokenizer',[sys.executable,'-B',package/helper,'--output',args.output/(model+'-tokenizer')],900)
        mg_wrapper=args.output/'MG-serve-stock.sh'
        run('MG-wrapper',[sys.executable,'-B',package/'make_mg_stock_quality_wrapper.py',
            '--model-path',args.MG_model,'--tokenizer-path',args.output/'MG-tokenizer/tokenizer',
            '--vllm-bin',args.native_venv/'bin/vllm' if args.backend=='vllm' else native,
            '--sglang-python',native,'--task-root',args.output/'MG-cache','--output',mg_wrapper,
            '--prefill-budget','4096','--use-default-memory-budget'],120)
        short=Path('/tmp')/('sb-default-MG-'+str(os.getpid()))
        short.mkdir()
        import shlex
        text=mg_wrapper.read_text()
        assert text.count('TMPDIR="$task/tmp"')==1
        mg_wrapper.write_text(text.replace('TMPDIR="$task/tmp"','TMPDIR='+shlex.quote(str(short))))
        gemma_wrapper=args.output/'Gemma-serve-stock.sh'
        write_wrapper(gemma_wrapper,'gemma4',args.backend,args.output/'Gemma-tokenizer/tokenizer',
                      args.Gemma_model,native,args.output/'Gemma-cache')
        for wrapper in (mg_wrapper,gemma_wrapper):
            assert '--gpu-memory-utilization' not in wrapper.read_text()
            assert '--mem-fraction-static' not in wrapper.read_text()
            run(wrapper.stem+'-syntax',['bash','-n',wrapper],30)
        tasks=[]
        for model,alias,wrapper,client,prefill in [
            ('MG','muse-glimmer-30b',mg_wrapper,'benchmark_mg.py',4096),
            ('Gemma','gemma4-12b-it',gemma_wrapper,'benchmark_gemma.py',2048)]:
            tasks.append(dict(name=model+'-'+args.backend,model=alias,backend=args.backend,
                wrapper=str(wrapper),client=str(package/client),inputs=str(inputs),
                benchmark_python=str(inputs/'.venv/bin/python'),prefill_budget=prefill,
                native_memory_budget='installed default; no explicit fraction',quality_policy='smoke',
                stock_framework_code_unmodified=True))
        config=args.output/'matrix-config.json'
        config.write_text(json.dumps(dict(tasks=tasks,port=args.port,lock=str(args.lock),
            performance_after_performance=str(args.after_performance) if args.after_performance else None,
            native_environment_freeze_SHA256=sha(args.output/'package-freeze.log')),indent=2)+'\n')
        state('running-single-round-matrix-before-deferred-quality',matrix_config=str(config))
        run('matrix',[sys.executable,'-B','-u',package/'run_default_chunk_single_matrix.py',
                      '--config',config,'--output',args.output/'matrix'],86400)
        state('stock-default-chunk-performance-and-quality-collected')
    except BaseException as error:
        state('failed-preserve-existing-results',error=repr(error),traceback=traceback.format_exc())
        raise


if __name__=='__main__':main()
