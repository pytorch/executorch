"""Collect unchanged full MG task correctness after mini and native full checks.

This collects raw F1 cases only. Full HTTP/performance qualification and main
acceptance are separate. Existing artifacts and evaluation data stay unchanged.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

from independent_session_full_native_binding import sha, verify_runtime
from prepare_independent_full_runtime_rebind import verify_rebind
from review_full_same_PTE_native_lifecycle import review
from prepare_mg_canonical_tokenizer import FILES


def verified(args, package):
    gate = verify_runtime(args.native_job, package)
    root = args.runtime_rebind_job
    state = json.loads((root/'progress.json').read_text())
    assert state['status']=='full-artifacts-ready-for-qualification'
    assert state['mini_gate_passed'] and state['full_profile']=='muse-glimmer-52layers'
    proof = verify_rebind(root/'full-runtime-rebind.json', args.native_job, package)
    alias = json.loads((root/'native-gate.json').read_text())
    assert alias['source']==gate['source']
    assert alias['full_runtime_rebind_proof_SHA256']==sha(root/'full-runtime-rebind.json')
    artifact = root/'full-independent-artifact'
    assert artifact.resolve()==Path(proof['artifacts']['independent']['artifact']).resolve()
    memory = json.loads(args.memory_worker_proof.read_text())
    assert memory['native_gate_SHA256']==sha(args.native_job/'native-gate.json')
    assert memory['source']==gate['source'] and memory['official_control_flow_preserved']
    assert memory['no_inference_kernel_change']
    assert memory['original_worker_SHA256']==sha(root/'model-build/muse_glimmer_batching_worker')
    worker=Path(memory['instrumented_worker'])
    assert sha(worker)==memory['instrumented_worker_SHA256']
    assert memory['optional_sampler_header_SHA256']==sha(package/'kv_memory_sampler.h')
    for name, expected in memory['unchanged_SDK_libraries'].items():assert sha(name)==expected
    for name,(expected,count) in FILES.items():
        path=args.tokenizer/name
        assert path.stat().st_size==count and sha(path)==expected,name
    source=Path(gate['source'])
    cache=(source/'extension/llm/cache/cache.h').read_text()
    assert 'int initial_capacity = 512;' in cache
    factory=(source/'examples/models/muse-glimmer/runtime/backend_factory.cpp').read_text()
    assert 'CudaExecutorConfig executor_config;' in factory
    explicit_initial='executor_config.initial_capacity' in factory
    if explicit_initial:
        assert 'executor_config.initial_capacity = config.initial_kv_capacity;' in factory
        assert '"--initial-kv-capacity"' in (source/'examples/models/muse-glimmer/serving/serve_batching.py').read_text()
    else:
        # The existing factory uses the verified CacheConfig default512.
        # Do not invent an unsupported server/worker argument.
        assert 'initial_capacity' not in factory
    graph=json.loads((args.full_graph_job/'actual-graph-independent-review.json').read_text())
    raw=json.loads((args.full_graph_job/'result.json').read_text())
    assert graph['actual_CUDA_trace'] and graph['all_seven_static_methods_after_shrink_verified']
    assert graph['automatic_reclaim_verified_without_compact_calls']
    assert graph['model_PTE_sha256']==raw['model_PTE_sha256']==sha(artifact/'model.pte')
    assert graph['probe_sha256']==raw['probe_sha256']==sha(root/'model-build/muse_glimmer_runtime_kv_probe')
    assert graph['sqlite_sha256']==sha(args.full_graph_job/'round-1/cuda-graphs.sqlite')
    return artifact,source,worker,explicit_initial


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['serve','quality'],default='quality')
    for name in ('package','runtime-rebind-job','native-job','before-full-capacity-job',
                 'full-capacity-job','full-graph-job','memory-worker-proof',
                 'tokenizer','public-inputs','lock','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--port',type=int,default=19916)
    parser.add_argument('--concurrencies',type=int,nargs='+',choices=[1,2,4,8,16],default=[16,1,8,4,2])
    parser.add_argument('--prefixes',nargs='+',choices=['off','on'],default=['off','on'])
    args=parser.parse_args();package=args.package.resolve()
    assert 1024<=args.port<=65535
    artifact,source,worker,explicit_initial=verified(args,package)
    if args.phase=='serve':
        prefix=os.environ['SB_PREFIX_CACHE'];assert prefix in ('off','on')
        port=int(os.environ['SB_PORT']);assert 1024<=port<=65535
        command=[sys.executable,'-B',str(source/'examples/models/muse-glimmer/serving/serve_batching.py'),
            '--worker-bin',str(worker),'--model-path',str(artifact/'model.pte'),
            '--data-path',str(artifact/'aoti_cuda_blob.ptd'),
            '--tokenizer-path',str(args.tokenizer),'--hf-tokenizer',str(args.tokenizer),
            '--model-id','muse-glimmer-30b','--backend','cuda',
            '--max-context','131072',
            '--max-sessions','16','--max-decode-sequences','16','--max-inflight-requests','16',
            '--prefix-cache-entries','4' if prefix=='on' else '0',
            '--prefix-checkpoint-interval','0','--tool-parser','none',
            '--host','127.0.0.1','--port',str(port)]
        if explicit_initial:command+=['--initial-kv-capacity','512']
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',
            PYTHONPATH=str(source.parent),ET_CUDA_BENCH_MEMORY_TRACE='1')
        print('FULL_MG_QUALITY_BINDING:'+json.dumps(dict(argv=command,
            native_gate_SHA256=sha(args.native_job/'native-gate.json'),
            model_PTE_SHA256=sha(artifact/'model.pte'),prefix=prefix,
            initial_KV_capacity=512,initial_KV_uses_existing_default=not explicit_initial,
            full_model_qualified=False,main_candidate_accepted=False)),flush=True)
        os.execvpe(command[0],command,env)
    assert not args.output.exists();args.output.mkdir(parents=True)
    def state(status,**extra):
        pending=args.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            full_model_qualified=False,main_candidate_accepted=False,**extra),indent=2)+'\n')
        pending.replace(args.output/'progress.json')
    try:
        state('independently-reviewing-full-same-PTE-native-lifecycle')
        result=review(args.before_full_capacity_job,args.full_capacity_job,'muse-glimmer-52layers')
        assert result['native_gate_SHA256'][1]==sha(args.runtime_rebind_job/'native-gate.json')
        assert result['full_FP32_tensors_bitwise_equal']==39 and result['forward_steps_per_arm']==1788
        (args.output/'full-native-lifecycle-review.json').write_text(json.dumps(result,indent=2)+'\n')
        deadline=time.monotonic()+1800
        while True:
            prep=json.loads((args.public_inputs/'progress.json').read_text())
            if prep['status']=='pinned-public-main-inputs-ready':break
            assert prep['status']!='failed',prep
            handle=Path('/proc')/str(prep['controller_pid'])/'cmdline'
            assert handle.is_file() and b'prepare_independent_main_public_inputs.py' in handle.read_bytes()
            assert time.monotonic()<deadline
            state('waiting-confirmed-live-pinned-public-inputs');time.sleep(20)
        benchmark_python=args.public_inputs/'.venv/bin/python'
        assert benchmark_python.is_file()
        common=[]
        for name in ('package','runtime-rebind-job','native-job','before-full-capacity-job',
                     'full-capacity-job','full-graph-job','memory-worker-proof',
                     'tokenizer','public-inputs','lock','output'):
            common+=['--'+name,str(getattr(args,name.replace('-','_')))]
        wrapper=args.output/'serve-qualified-full-MG.sh'
        wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
            'test "${1:?backend}" = executorch\nexec '+shlex.join(
                [sys.executable,'-B',str(Path(__file__).resolve()),'--phase','serve',*common])+'\n')
        subprocess.run(['bash','-n',str(wrapper)],check=True)
        completed=[]
        for prefix in args.prefixes:
            for concurrency in args.concurrencies:
                scope='full' if concurrency in (1,16) else 'smoke'
                case=args.output/('correctness-agent-prefix-'+prefix+'-c'+str(concurrency))
                command=[str(benchmark_python),'-B',str(package/'run_case_24h.py'),
                    '--wrapper',str(wrapper),'--backend','executorch','--model','muse-glimmer-30b',
                    '--variant','independent-session-KV','--prefix-cache',prefix,
                    '--client',str(package/'benchmark_mg.py'),
                    '--prompts',str(args.public_inputs/'agent_benchmark_prompts.json'),
                    '--correctness-root',str(args.public_inputs/'correctness'),
                    '--python',str(benchmark_python),'--concurrency',str(concurrency),'--port',str(args.port),
                    '--output',str(case),'--lock',str(args.lock),'--mode','correctness',
                    '--split','agent','--quality-scope',scope,'--correctness-rounds','1',
                    '--correctness-max-tokens','1024','--correctness-chat-template-kwargs',
                    json.dumps(dict(enable_thinking=False,current_date='2026-10-21',reasoning_strength='low'))]
                state('collecting-full-MG-task-correctness',prefix=prefix,concurrency=concurrency,
                    quality_scope=scope,completed=completed)
                with (args.output/(case.name+'-controller.log')).open('x') as log:
                    subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,
                        stdin=subprocess.DEVNULL,timeout=21600)
                completed.append(dict(prefix=prefix,concurrency=concurrency,scope=scope,case=str(case)))
        (args.output/'result.json').write_text(json.dumps(dict(cases=completed,
            native_runtime_gate_SHA256=sha(args.native_job/'native-gate.json'),
            full_native_review_SHA256=sha(args.output/'full-native-lifecycle-review.json'),
            full_task_F1_independent_review_pending=True,full_model_qualified=False,
            main_candidate_accepted=False),indent=2)+'\n')
        state('full-MG-task-correctness-collected-pending-independent-review',completed=completed)
    except Exception as error:
        state('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
