"""Isolated current-artifact c8/c16 diagnostic; preserve accepted main results."""
import argparse,datetime,hashlib,json,os,shlex,signal,subprocess,sys,time,traceback
from pathlib import Path
PTE_SHA='2d627d22c91bd284c981cde8de88d83c083b7e002c454582cab2edccb3071678'
PTD_SHA='861d6eef785d4ce4368a99dfc0458bf24d0b92134ab5218733dc4ff0611f20f8'
WORKER_SHA='706b46ed2240c3c4306f4210cf21269333913ffe539768c278b92495187b4d1a'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['origin','source','artifact','python','benchmark-python','output','lock']:p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--phase',choices=['collect','serve','profile'],default='collect');p.add_argument('--port',type=int,default=19627)
 a=p.parse_args();pkg=Path(__file__).resolve().parent.parent
 if a.phase=='collect':a.output.mkdir(exist_ok=False,parents=True)
 def state(s,**kw):
  t=a.output/'progress.pending.json';t.write_text(json.dumps(dict(status=s,PID=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_main_remains_paused=True,**kw),indent=2)+'\n');t.replace(a.output/'progress.json')
 assert sha(a.artifact/'model.pte')==PTE_SHA
 assert sha(a.origin/'runtime/gemma_runtime_columns_worker')==WORKER_SHA
 # The 8 GB weights were independently verified during receiving. Check receipt,
 # inode and size on each launch; no concurrent mutation of this read-only file.
 r=json.loads((a.artifact/'transfer-provenance.json').read_text());assert r['PTD_SHA256']==PTD_SHA and r['PTE_SHA256']==PTE_SHA
 assert os.stat(a.artifact/'aoti_cuda_blob.ptd').st_ino==os.stat(a.origin/'artifact/aoti_cuda_blob.ptd').st_ino
 assert (a.artifact/'aoti_cuda_blob.ptd').stat().st_size==8017694976
 command=[str(a.python),'-B',str(pkg/'serve_gemma_runtime_columns.py'),'--worker-bin',str(a.origin/'runtime/gemma_runtime_columns_worker'),'--model-path',str(a.artifact/'model.pte'),'--data-path',str(a.artifact/'aoti_cuda_blob.ptd'),'--tokenizer-path',str(a.origin/'tokenizer'),'--hf-tokenizer',str(a.origin/'tokenizer'),'--model-id','gemma4-12b-it','--max-context','131072','--initial-kv-capacity','512','--max-sessions','16','--max-decode-sequences','16','--max-inflight-requests','16','--prefix-cache-entries','1' if os.environ.get('SB_PREFIX_CACHE')=='on' else '0','--prefix-checkpoint-interval','512' if os.environ.get('SB_PREFIX_CACHE')=='on' else '0','--tool-parser','none','--raw-tool-text','--host','127.0.0.1','--port',str(int(os.environ.get('SB_PORT',a.port)))]
 env=dict(os.environ,PYTHONPATH=str(a.source.parent)+os.pathsep+os.environ.get('PYTHONPATH',''),LD_LIBRARY_PATH=str(a.origin/'runtime')+os.pathsep+os.environ.get('LD_LIBRARY_PATH',''),CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',ET_CUDA_BENCH_MEMORY_TRACE='1')
 if a.phase=='serve':
  print('SCALING_DIAGNOSTIC_BINDING:'+json.dumps(dict(argv=command,PTE_SHA256=PTE_SHA,PTD_SHA256=PTD_SHA,worker_SHA256=WORKER_SHA,unchanged_runtime=True,unchanged_model_methods=True,main_results_not_replaced=True,monotonic_ns=time.monotonic_ns())),flush=True);os.execvpe(command[0],command,env)
 if a.phase=='profile':
  case=Path(os.environ['SCALING_PROFILE_CASE']);trace=case/'graph-trace'
  argv=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace=graph','--cuda-flush-interval=1000','--wait=all','--output',str(trace),*command]
  child=subprocess.Popen(argv,env=env,start_new_session=True)
  def stop(*_):
   # Give the traced Python server its own graceful shutdown. The profiler
   # stays alive to flush the complete trace and finalize its report.
   found=[]
   for d in Path('/proc').iterdir():
    if not d.name.isdigit():continue
    try:
     cmd=(d/'cmdline').read_bytes().split(b'\0');stat=(d/'stat').read_text().rsplit(') ',1)[1].split()
     if int(stat[3])==child.pid and str(pkg/'serve_gemma_runtime_columns.py').encode() in cmd:found.append(int(d.name))
    except (FileNotFoundError,ProcessLookupError,PermissionError):pass
   for pid in found:os.kill(pid,signal.SIGINT)
   if not found:os.killpg(child.pid,signal.SIGINT)
  signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
  rc=child.wait();(case/'profile-finalization.json').write_text(json.dumps(dict(returncode=rc,nsys_argv=argv,reports=[str(f) for f in case.glob('graph-trace*')],shutdown_graceful_requested=True),indent=2)+'\n');sys.exit(0 if rc==0 else 1)
 completed=[]
 try:
  state('checking-current-runtime-and-artifact')
  common=['--origin',str(a.origin),'--source',str(a.source),'--artifact',str(a.artifact),'--python',str(a.python),'--benchmark-python',str(a.benchmark_python),'--output',str(a.output),'--lock',str(a.lock),'--port',str(a.port)]
  for profile in [False,True]:
   for split,prefix in [('chat','on'),('agent','off')]:
    for c in [8,16]:
     case=a.output/(('profile-' if profile else 'unprofiled-')+split+'-'+prefix+'-c'+str(c))
     wrapper=a.output/(case.name+'-serve.sh')
     wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\ntest "${1:?}" = executorch\n'+('export SCALING_PROFILE_CASE='+shlex.quote(str(case))+'\n' if profile else '')+'exec '+shlex.join([str(a.python),'-B',str(Path(__file__).resolve()),'--phase','profile' if profile else 'serve',*common])+'\n')
     argv=[str(a.benchmark_python),'-B',str(pkg/'run_case_24h.py'),'--wrapper',str(wrapper),'--backend','executorch','--model','gemma4-12b-it','--variant','current2048-unchanged-runtime-scaling-diagnostic','--prefix-cache',prefix,'--client',str(pkg/'benchmark_gemma.py'),'--prompts',str(a.origin/'inputs'/(split+'_benchmark_prompts.json')),'--correctness-root',str(a.origin/'inputs/correctness'),'--python',str(a.benchmark_python),'--concurrency',str(c),'--port',str(a.port),'--output',str(case),'--lock',str(a.lock),'--mode','performance','--split',split,'--performance-rounds','1']
     state('collecting-bounded-profile' if profile else 'collecting-unprofiled-control',case=str(case),completed=completed)
     (a.output/(case.name+'-command.json')).write_text(json.dumps(dict(argv=argv,profile=profile),indent=2)+'\n')
     with (a.output/(case.name+'-controller.log')).open('x') as f:subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=2400)
     if profile:
      report=case/'graph-trace.nsys-rep'
      assert report.is_file() and report.stat().st_size>0,'Profiler report missing; no idle-time inference permitted'
      with (case/'nsys-export.log').open('x') as f:subprocess.run(['nsys','export','--type','sqlite','--output',str(case/'graph-trace.sqlite'),str(report)],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
     completed.append(case.name)
  state('collected-pending-GPU-span-and-queue-review',completed=completed,fixed_length_control_pending=True,TTFT_formula_alone_not_used_to_explain_throughput=True)
 except Exception as e:
  (a.output/'exception.txt').write_text(traceback.format_exc());state('failed-preserve-evidence',error_type=type(e).__name__,detail=str(e)[-1500:],completed=completed);raise
if __name__=='__main__':main()
