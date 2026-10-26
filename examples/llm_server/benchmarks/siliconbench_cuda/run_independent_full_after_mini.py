"""Wait for actual mini checks, independently review them, then export full model.

The full export stays disabled on any unresolved mini timing/correctness signal.
Export completion only prepares a candidate for full-model qualification and
the main rerun. No old source, result, model or GitHub content is deleted.
"""
import argparse,datetime,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('mini-job','native-job','previous-latency-job','gguf','output','lock'):
  p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--profile',choices=['muse-glimmer-first4','gemma4-first6'],required=True)
 p.add_argument('--source-preparation-job',type=Path)
 p.add_argument('--original-export-source',type=Path)
 p.add_argument('--original-export-native-gate',type=Path)
 p.add_argument('--Gemma-global-baseline',action='store_true',
                help='Prepare a separate global-row same-GPU baseline after the same mini gate')
 a=p.parse_args();gemma=a.profile=='gemma4-first6'
 assert not a.Gemma_global_baseline or gemma
 assert a.gguf.is_file() and not a.output.exists()
 if gemma:assert a.source_preparation_job
 else:assert a.original_export_source and a.original_export_native_gate
 a.output.mkdir(parents=True);package=Path(__file__).resolve().parent
 def state(status,**extra):
  tmp=a.output/'progress.pending.json'
  tmp.write_text(json.dumps({'status':status,'controller_pid':os.getpid(),
   'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'full_model_qualified':False,
   'main_candidate_accepted':False,**extra},indent=2)+'\n');tmp.replace(a.output/'progress.json')
 def run(label,script,args):
  command=[sys.executable,'-B','-u',str(package/script),*map(str,args)]
  (a.output/(label+'-command.json')).write_text(json.dumps({'argv':command},indent=2)+'\n')
  with (a.output/(label+'.log')).open('x') as log:
   subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                  check=True,timeout=21600)
 try:
  deadline=time.monotonic()+120*60
  while True:
   d=json.loads((a.mini_job/'progress.json').read_text())
   if d['status']=='growth-allocator-mini-controls-collected-pending-independent-review':break
   assert d['status']!='failed',d
   proc=Path('/proc')/str(d['controller_pid'])/'cmdline'
   assert proc.is_file() and ('run_growth_allocator_'+('gemma' if gemma else 'mg')+'_mini_controls.py').encode() in proc.read_bytes()
   assert time.monotonic()<deadline,'Verified-live observation deadline; preserve the existing mini job'
   state('waiting-confirmed-live-mini-control',mini_PID=d['controller_pid'],mini_status=d['status'],full_model_export_started=False)
   time.sleep(15)
  state('independently-reviewing-mini-raw-evidence',full_model_export_started=False)
  qargs=['--mini-job',a.mini_job,'--native-job',a.native_job,
         '--previous-latency-job',a.previous_latency_job,'--profile',a.profile,'--output',a.output/'mini-reviews']
  if gemma:qargs+=['--Gemma-source-preparation-job',a.source_preparation_job]
  run('mini-raw-review','qualify_independent_session_mini_for_full.py',qargs)
  qpath=a.output/'mini-reviews/mini-qualification.json';q=json.loads(qpath.read_text())
  if not q['full_model_export_for_qualification_allowed']:
   state('mini-performance-gate-not-passed',positive_performance_signals=q['positive_performance_signals'],full_model_export_started=False)
   return
  assert q['mini_storage_correctness_pass'] and q['mini_cold_and_steady_performance_pass'] and q['mini_HTTP_memory_pass']
  required=20 if gemma else 35
  assert shutil.disk_usage(a.output).free>required*2**30,'Insufficient isolated output space; preserve existing files'
  state('mini-qualified-starting-full-export-preparation',full_model_export_started=True,
        full_layers=48 if gemma else 52,mini_qualification=str(qpath))
  common=['--gguf',a.gguf,'--mini-qualification',qpath,'--runtime-native-job',a.native_job,'--lock',a.lock]
  if gemma:
   layout='global' if a.Gemma_global_baseline else 'independent'
   run('full-export','prepare_gemma_independent_session_full.py',
       ['--source-preparation-job',a.source_preparation_job,'--output',a.output/'full-export','--layout',layout,*common])
   artifact=a.output/('full-export/full-'+layout+'-artifact')
  else:
   run('full-export','export_mg_independent_session_full.py',
       ['--source',a.original_export_source,'--native-gate',a.original_export_native_gate,
        '--output',a.output/'full-independent-artifact','--job-dir',a.output/'full-export',*common])
   artifact=a.output/'full-independent-artifact'
  assert (artifact/'independent-full-artifact-audit.json').is_file()
  state('full-artifact-ready-pending-full-model-qualification',full_model_export_started=True,
        artifact=str(artifact),full_layers=48 if gemma else 52,
        remaining_gates=['full-model correctness/F1','same-GPU fixed-token performance','automatic KV reclaim and memory','affected main rerun'])
 except Exception as error:
  state('failed',error=str(error),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
