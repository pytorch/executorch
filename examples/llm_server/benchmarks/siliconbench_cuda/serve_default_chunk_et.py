"""Serve a new prefill-ceiling artifact with the already qualified runtime.

The previous wrapper supplies the verified native, dependency, tokenizer and
allocator bindings. Only the model artifact is replaced. The new artifact has
its own audit and subsequent task quality checks; prior artifact qualification
is not claimed for the replacement.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import time


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-wrapper',type=Path,required=True)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--prefill-budget',type=int,choices=(2048,4096),required=True)
    args=parser.parse_args()
    line=next(line for line in args.old_wrapper.read_text().splitlines() if line.startswith('exec '))
    previous=shlex.split(line)[1:]
    def value(flag):return previous[previous.index(flag)+1]
    package=Path(value('--package'))
    sys.path.insert(0,str(package))
    from independent_session_full_native_binding import sha
    from independent_session_full_http_binding import verified_gemma_full_http,verified_mg_full_http
    profile=value('--full-model-profile')
    gemma=profile=='gemma4-48layers'
    a=argparse.Namespace()
    a.full_model_profile=profile
    for flag in ('HTTP-job','native-job','memory-worker-proof','http-python-dependencies-job',
                 'tokenizer','server-python','public-inputs','lock','output',
                 *(('fixed-job','Gemma-source-preparation-job') if gemma else ('native-lifecycle-job','mini-qualification'))):
        setattr(a,flag.replace('-','_'),Path(value('--'+flag)))
    a.mini_profile='gemma4-first6' if gemma else 'muse-glimmer-first4'
    a.full_native_job=a.native_job
    a.preparation_job=a.fixed_job/'full-pair' if gemma else a.native_lifecycle_job/'after-full-artifact'
    a.full_MG_native_lifecycle_job=None if gemma else a.native_lifecycle_job
    a.full_MG_mini_qualification=None if gemma else a.mini_qualification
    a.arm='independent'
    a.preserved_global_HTTP_failure_job=None
    a.disable_native_memory_sampling=False
    binding=(verified_gemma_full_http if gemma else verified_mg_full_http)(a,package)
    audit=json.loads((args.artifact/'independent-full-artifact-audit.json').read_text())
    assert audit['independent_capacity'] and audit['initial_runtime_KV']==512
    assert audit['logical_context']==131072 and audit['selected_logit_limit']==64
    assert audit['metadata']['get_max_prefill_chunk']==args.prefill_budget
    assert audit['GGUF_sha256']==('7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a' if gemma else
        '4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e')
    for name,proof in audit['artifacts'].items():assert sha(args.artifact/name)==proof['sha256']
    prefix=os.environ['SB_PREFIX_CACHE'];assert prefix in ('off','on')
    port=int(os.environ['SB_PORT']);assert 1024<=port<=65535
    serve=package/'serve_gemma_runtime_columns.py' if gemma else binding['source']/'examples/models/muse-glimmer/serving/serve_batching.py'
    command=[str(a.server_python),'-B',str(serve),'--worker-bin',str(binding['worker']),
        '--model-path',str(args.artifact/'model.pte'),'--data-path',str(args.artifact/'aoti_cuda_blob.ptd'),
        '--tokenizer-path',str(a.tokenizer),'--hf-tokenizer',str(a.tokenizer),
        '--model-id','gemma4-12b-it' if gemma else 'muse-glimmer-30b','--max-context','131072',
        '--max-sessions','16','--max-decode-sequences','16','--max-inflight-requests','16',
        '--prefix-cache-entries',('1' if gemma else '4') if prefix=='on' else '0',
        '--prefix-checkpoint-interval','512' if gemma and prefix=='on' else '0',
        '--tool-parser','none','--host','127.0.0.1','--port',str(port)]
    if binding['explicit_initial']:command+=['--initial-kv-capacity','512']
    command+=['--raw-tool-text'] if gemma else ['--backend','cuda']
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',
        ET_CUDA_BENCH_MEMORY_TRACE='1',
        PYTHONPATH=str(binding['dependency_path'])+os.pathsep+str(binding['source'].parent))
    print('FULL_MAIN_SERVER_BINDING:'+json.dumps(dict(argv=command,full_profile=profile,
        model_PTE_SHA256=sha(args.artifact/'model.pte'),native_runtime_gate_SHA256=sha(a.native_job/'native-gate.json'),
        memory_worker_SHA256=sha(binding['worker']),prefix=prefix,static_forward_graphs_enabled=True,
        legacy_idle_policy_unchanged=True,server_binding_monotonic_ns=time.monotonic_ns(),
        initial_KV=512,logical_context=131072,prefill_budget=args.prefill_budget,
        prior_native_qualification_reused=True,new_artifact_task_quality_pending=True)),flush=True)
    os.execvpe(command[0],command,env)


if __name__=='__main__':main()
