"""Add general static128/256/512 methods, preserve64-row heads, rerun roofline.

Use one isolated source worktree and official SDK. No72/520-specific methods.
Original main artifacts and results remain inputs, never mutable destinations.
CPU SDK build overlaps the single locked GPU export; all jobs survive SSH exit.
"""
import argparse,datetime,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path

PUBLIC_HEAD='336d58a73fcc8eaabdbaa171f237fcf4f61755ad'
STATIC=[1,2,4,8,16,32,64,128,256,512]
SOURCE_HASHES={
 'backends/cuda/batching/cuda_executor.cpp':'4db6d9f768fb2c6d280321ebbb1b624a7182527ccb5cca97215d222a93658e4c',
 'backends/cuda/batching/cuda_executor.h':'ea130a8a59413775e100c23aa779daad070c60a0f91acbef1500f7f69162bfcf',
 'examples/models/muse-glimmer/export/export_solo.py':'d0a00dc34ae219de6879b985586d7fc83693d4340f1765a4bd3490361de303ec'}
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(16*1024**2):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['root','et-source','gguf','baseline-export-job','baseline-artifact','CPP-SDK-job','validated-policy-file','reference-deployment','slice-copy-recovery-job','lock','output']:
  p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--attention-source-sha256',required=True)
 p.add_argument('--deadline-UTC',required=True)
 p.add_argument('--serialization-recovery-workflow',type=Path)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 deadline=datetime.datetime.fromisoformat(a.deadline_UTC);assert deadline.tzinfo
 package=Path(__file__).resolve().parent;children=[]
 def remaining():
  t=(deadline-datetime.datetime.now(datetime.timezone.utc)).total_seconds();assert t>0,'Preserve outputs at deadline';return t
 def state(status,**extra):
  f=a.output/'progress.pending.json';f.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),deadline_UTC=a.deadline_UTC,static_methods=STATIC,no_workload_specific_methods=True,main_results_changed=False,**extra),indent=2)+'\n');f.replace(a.output/'progress.json')
 def start(command,label):
  log=(a.output/(label+'.log')).open('x')
  child=subprocess.Popen([str(v) for v in command],stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();children.append(child)
  return child
 def finish(child,label):
  state('running-'+label,child_pid=child.pid);assert child.wait(timeout=remaining())==0,label
 def run(command,label):finish(start(command,label),label)
 try:
  state('preparing-isolated-public-source')
  assert subprocess.check_output(['git','-C',str(a.et_source),'rev-parse','HEAD'],text=True).strip()==PUBLIC_HEAD
  for name,digest in SOURCE_HASHES.items():assert sha(a.et_source/name)==digest
  source=a.output/'source/executorch';source.parent.mkdir()
  subprocess.run(['git','-C',str(a.et_source),'worktree','add','--detach',str(source),PUBLIC_HEAD],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
  # Reuse existing exact-version dependency checkouts as read-only source
  # links; all build output is directed to this workflow's separate directory.
  config=subprocess.run(['git','-C',str(a.et_source),'config','-f','.gitmodules','--get-regexp',r'submodule\..*\.path'],capture_output=True,text=True)
  for line in config.stdout.splitlines():
   relative=line.split(maxsplit=1)[1];old=a.et_source/relative;new=source/relative
   if old.is_dir() and any(old.iterdir()):
    if new.exists():new.rmdir()
    new.parent.mkdir(parents=True,exist_ok=True);new.symlink_to(old,target_is_directory=True)
  patch=package/'roofline_static_selector_rows.patch'
  subprocess.run(['git','-C',str(source),'apply','--check',str(patch)],check=True)
  subprocess.run(['git','-C',str(source),'apply',str(patch)],check=True)
  resources={}
  for name in ['program.fbs','scalar_type.fbs']:
   original=source/'schema'/name;target=source/'exir/_serialize'/name
   assert sha(original)==sha(a.et_source/'schema'/name)
   assert not target.exists()
   target.write_bytes(original.read_bytes());resources[name]=sha(target)
  flatc=a.et_source/'exir/_serialize/flatbuffers-flatc'
  if flatc.is_file():
   target=source/'exir/_serialize/flatbuffers-flatc';target.write_bytes(flatc.read_bytes());target.chmod(flatc.stat().st_mode)
   resources['flatbuffers-flatc']=sha(target)
  else:
   import shutil
   assert shutil.which('flatc'),'Serialization compiler must exist before GPU compilation'
  (a.output/'serialization-resource-provenance.json').write_text(json.dumps(resources,indent=2)+'\n')
  (a.output/'source-patch-provenance.json').write_text(json.dumps(dict(public_HEAD=PUBLIC_HEAD,patch_SHA256=sha(patch),original_source_SHA256=SOURCE_HASHES,patched_source_SHA256={n:sha(source/n) for n in SOURCE_HASHES},static_methods=STATIC,maximum_selected_rows=64,original_source_unchanged=True),indent=2)+'\n')
  env_ready=a.output/'environment-ready.json';env_ready.write_text(json.dumps(dict(phases=dict(environment='ready')))+'\n')
  sdk=a.output/'native-SDK';sdk_child=start([sys.executable,'-B',package/'prepare_roofline_gpu_sdk.py','--root',a.root,'--et-source',source,'--python',sys.executable,'--setup-progress',env_ready,'--output',sdk,'--only-ET'],'native-SDK-build')
  policies=json.loads(a.validated_policy_file.read_text());assert len(policies['policies'])==len(policies['validation_reports'])==3
  combined=a.output/'qualified-policies.json';combined.write_text(json.dumps(policies,indent=2)+'\n')
  artifact=a.output/'candidate-artifact';export=a.output/'candidate-export'
  export_command=[sys.executable,'-B',package/'export_mg_roofline.py','--et-source',source,'--gguf',a.gguf,'--reference-deployment',a.reference_deployment,'--attention-source-sha256',a.attention_source_sha256,'--forward-sizes',','.join(map(str,STATIC)),'--large-w4a16-policy-file',combined,'--slice-copy-recovery-job',a.slice_copy_recovery_job,'--reference-export-job',a.baseline_export_job,'--output',artifact,'--job-dir',export,'--lock',a.lock]
  if a.serialization_recovery_workflow:
   previous=json.loads((a.serialization_recovery_workflow/'progress.json').read_text())
   assert previous['status']=='failed' and not (Path('/proc')/str(previous['controller_pid'])).exists()
   old=json.loads((a.serialization_recovery_workflow/'source-patch-provenance.json').read_text())
   assert old['patched_source_SHA256']=={n:sha(source/n) for n in SOURCE_HASHES}
   export_command += ['--serialization-recovery-job',a.serialization_recovery_workflow/'candidate-export']
  export_child=start(export_command,'static-full-model-export')
  state('SDK-building-and-static-forward-export-running',SDK_controller_pid=sdk_child.pid,export_controller_pid=export_child.pid,isolated_source=str(source))
  assert export_child.wait(timeout=remaining())==0,'static-full-model-export'
  finish(sdk_child,'native-SDK-build')
  audit=json.loads((artifact/'artifact-audit.json').read_text());assert audit['static_methods']==STATIC and all(v[0]<=64 for v in audit['static_output_bounds'].values())
  assert sha(artifact/'aoti_cuda_blob.ptd')==sha(a.baseline_artifact/'aoti_cuda_blob.ptd'),'Same official quantized weight bytes required'
  (sdk/'source-overlay-provenance.json').write_text((a.output/'source-patch-provenance.json').read_text())
  arms=a.output/'comparison-arms.json';arms.write_text(json.dumps(dict(ET_arms=[dict(name='ET-baseline',artifact=str(a.baseline_artifact),export_job=str(a.baseline_export_job)),dict(name='ET-large',artifact=str(artifact),export_job=str(export))]),indent=2)+'\n')
  comparison=a.output/'full-model-comparison'
  run([sys.executable,'-B',package/'run_roofline_default_comparison.py','--ET-SDK-job',sdk,'--CPP-SDK-job',a.CPP_SDK_job,'--gguf',a.gguf,'--arms',arms,'--job',comparison,'--lock',a.lock,'--repetitions','20'],'three-arm-four-round-roofline')
  run([sys.executable,'-B',package/'profile_roofline_large_candidate.py','--et-source',source,'--ET-SDK-job',sdk,'--python',sys.executable,'--artifact',artifact,'--after-comparison',comparison,'--lock',a.lock,'--job',a.output/'candidate-profile'],'static-routing-and-CUDA-graph-profile')
  state('comparison-and-profile-completed-awaiting-validation',comparison=str(comparison),artifact=str(artifact))
  run([sys.executable,'-B',package/'validate_roofline_prefill_outputs.py','--workflow',a.output,'--ET-SDK-job',sdk,'--et-source',source,'--python',sys.executable,'--lock',a.lock,'--output',a.output/'output-validation','--deadline-UTC',a.deadline_UTC],'full-prefill-and-followup-output-validation')
  run([sys.executable,'-B',package/'run_roofline_history_logits_control.py','--workflow',a.output,'--ET-SDK-job',sdk,'--et-source',source,'--lock',a.lock,'--output',a.output/'zero-history-control','--history-tokens','0','--deadline-UTC',a.deadline_UTC],'teacher-forced-zero-history-control')
  state('completed-pending-independent-review',comparison=str(comparison),artifact=str(artifact),candidate_not_automatically_promoted=True)
 except BaseException as exc:
  state('failed',error=str(exc));raise
 finally:
  for child in children:
   if child.poll() is None:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=60)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
if __name__=='__main__':main()
