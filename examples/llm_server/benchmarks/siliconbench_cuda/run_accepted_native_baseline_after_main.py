"""Collect a same-GPU accepted native/artifact baseline after new main collection.

The preserved old server entry point uses the qualified current HTTP dependency
source. This controlled native/artifact comparison is not an exact historical
Python server reproduction. Keep its raw correctness and performance separate.
"""
import argparse,datetime,hashlib,json,os,shlex,subprocess,sys,time,traceback
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binding(config):
    stage=Path(config['stage']);manifest=json.loads((stage/'runtime-manifest.json').read_text())
    for original,record in manifest['files'].items():
        path=stage/'payload'/original.lstrip('/')
        assert path.resolve().is_relative_to(stage.resolve())
        assert path.stat().st_size==record['bytes'] and sha(path)==record['SHA256']
    result={key:Path(config[key]) for key in ('worker','PTE','PTD','serve_script','tokenizer','python_source','dependency_path','server_python','inputs','package','lock')}
    for key in ('worker','PTE','serve_script'):
        path=result[key];original='/'+path.relative_to(stage/'payload').as_posix()
        assert original in manifest['files']
    assert sha(result['PTD'])==config['old_PTD_SHA256']
    assert sha(result['serve_script'])==config['old_serve_script_SHA256']
    assert sha(result['worker'])==config['old_worker_SHA256'] and sha(result['PTE'])==config['old_PTE_SHA256']
    assert all(result[key].exists() for key in result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['collect','serve'],required=True)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--parent-main-job',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();config=json.loads(a.config.read_text())
    assert config['model'] in ('muse-glimmer-30b','gemma4-12b-it')
    gemma=config['model']=='gemma4-12b-it'
    if a.phase=='serve':
        b=binding(config);prefix=os.environ['SB_PREFIX_CACHE'];assert prefix in ('on','off')
        argv=[str(b['server_python']),'-B',str(b['serve_script']),
            '--worker-bin',str(b['worker']),'--model-path',str(b['PTE']),'--data-path',str(b['PTD']),
            '--tokenizer-path',str(b['tokenizer']/'tokenizer.json'),'--hf-tokenizer',str(b['tokenizer']),
            '--model-id',config['model'],'--max-context','131072','--initial-kv-capacity','512',
            '--max-sessions','16','--max-decode-sequences','16','--max-inflight-requests','16',
            '--prefix-cache-entries',('1' if gemma else '4') if prefix=='on' else '0',
            '--prefix-checkpoint-interval','512' if gemma and prefix=='on' else '0',
            '--host','127.0.0.1','--port',os.environ['SB_PORT']]
        argv+=['--raw-tool-text','--tool-parser','none'] if gemma else ['--backend','cuda']
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',
            PYTHONPATH=str(b['dependency_path'])+os.pathsep+str(b['python_source'].parent),
            LD_LIBRARY_PATH=os.pathsep.join(config['LD_library_paths']))
        env.pop('ET_CUDA_BENCH_MEMORY_TRACE',None)
        print('ACCEPTED_NATIVE_BASELINE_BINDING:'+json.dumps(dict(argv=argv,config_SHA256=sha(a.config),
            same_current_HTTP_dependency_source=True,exact_historical_Python_reproduction=False,
            original_worker_and_PTE=True,original_PTD_SHA256=config['old_PTD_SHA256'])),flush=True)
        os.execvpe(argv[0],argv,env)
    assert not a.output.exists();a.output.mkdir(parents=True)
    def state(status,**extra):
        p=a.output/'progress.pending.json';p.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),baseline_accepted=False,**extra),indent=2)+'\n');p.replace(a.output/'progress.json')
    try:
        deadline=time.monotonic()+36*3600
        while True:
            d=json.loads((a.parent_main_job/'progress.json').read_text())
            if d['status'] in ('qualified-MG-main-collected-pending-independent-review','qualified-Gemma-main-collected-pending-independent-review'):break
            assert d['status']!='failed',d
            cmd=Path('/proc')/str(d['controller_pid'])/'cmdline'
            assert cmd.is_file() and any(n in cmd.read_bytes() for n in (b'run_full_MG_main_after_fresh_quality.py',b'run_full_Gemma_main_after_fresh_quality.py'))
            assert time.monotonic()<deadline,'Preserve live parent; baseline observation deadline reached'
            state('waiting-live-new-main-before-old-baseline',parent_PID=d['controller_pid'],parent_status=d['status']);time.sleep(20)
        parent=json.loads((a.parent_main_job/'result.json').read_text());assert parent['main_cases_collected']==20
        b=binding(config);package=b['package'];inputs=b['inputs'];python=inputs/'.venv/bin/python'
        wrapper=a.output/'serve-old-native.sh'
        wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\ntest "${1:?}" = executorch\nexec '+shlex.join([str(b['server_python']),'-B',str(Path(__file__).resolve()),'--phase','serve','--config',str(a.config),'--parent-main-job',str(a.parent_main_job),'--output',str(a.output)])+'\n')
        subprocess.run(['bash','-n',str(wrapper)],check=True)
        (a.output/'provenance.json').write_text(json.dumps(dict(config=config,config_SHA256=sha(a.config),
            new_main_result_SHA256=sha(a.parent_main_job/'result.json'),same_current_HTTP_dependency_source=True,
            exact_historical_Python_reproduction=False,baseline_accepted=False),indent=2)+'\n')
        completed=[]
        for mode in ('correctness','performance'):
            for split in (('agent',) if mode=='correctness' else ('chat','agent')):
                for prefix in ('off','on'):
                    for c in ((16,1) if mode=='correctness' else (16,1,8,4,2)):
                        case=a.output/(mode+'-'+split+'-'+prefix+'-c'+str(c))
                        argv=[str(python),'-B',str(package/'run_case_24h.py'),'--wrapper',str(wrapper),
                            '--backend','executorch','--model',config['model'],'--variant','accepted-native-artifact-current-HTTP',
                            '--prefix-cache',prefix,'--client',str(package/('benchmark_gemma.py' if gemma else 'benchmark_mg.py')),
                            '--prompts',str(inputs/(split+'_benchmark_prompts.json')),'--correctness-root',str(inputs/'correctness'),
                            '--python',str(python),'--concurrency',str(c),'--port',str(config['port']),
                            '--output',str(case),'--lock',str(b['lock']),'--mode',mode,'--split',split]
                        if mode=='correctness':
                            argv+=['--quality-scope','smoke','--correctness-rounds','1']
                            if not gemma:argv+=['--correctness-max-tokens','1024','--correctness-chat-template-kwargs',json.dumps(dict(enable_thinking=False,current_date='2026-10-21',reasoning_strength='low'))]
                        state('collecting-same-GPU-old-native-baseline',case=str(case),completed=completed)
                        (a.output/(case.name+'-command.json')).write_text(json.dumps(dict(argv=argv),indent=2)+'\n')
                        with (a.output/(case.name+'.log')).open('x') as log:subprocess.run(argv,check=True,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=43200)
                        assert json.loads((case/'progress.json').read_text())['status']=='completed'
                        if mode=='correctness':
                            scores=json.loads((case/'correctness-scores.json').read_text());assert len(scores)==2
                            assert all(s['n_scored']==64 and s['request_errors']==0 and s['n_missing_responses']==s['n_extra_responses']==0 for s in scores)
                        else:
                            result=json.loads((case/'performance-aggregate.json').read_text());assert result['successful']['mean']==100 and result['failed']['mean']==0
                        completed.append(case.name)
        assert len(completed)==24
        (a.output/'result.json').write_text(json.dumps(dict(completed=completed,baseline_accepted=False,raw_review_pending=True,
            exact_historical_Python_reproduction=False,same_GPU_old_native_artifact_comparison_collected=True),indent=2)+'\n')
        state('old-native-baseline-collected-pending-raw-review',completed=completed)
    except Exception as error:state('failed',error=str(error),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
