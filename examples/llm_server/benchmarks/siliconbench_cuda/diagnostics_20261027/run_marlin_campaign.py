"""Download pinned public GGUFs and run independent original-Marlin probes."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time,traceback,urllib.request
from pathlib import Path
MODELS=[('Gemma','unsloth/gemma-4-12b-it-GGUF','fc034cfff751157913579611efad8462ac1be606','gemma-4-12b-it-Q4_0.gguf','7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a'),('MG','meta-models/Muse-Glimmer-30B-GGUF','b44f1072cf75e918a1b82a3a17b45c4389b076e7','Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf','4cc57c0f51040a226e5a72cc47b7613f7772950e460a665f7083de89f183f60e')]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ['source','python','output','lock']:p.add_argument('--'+n,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);completed=[]
 def state(s,**kw):
  t=a.output/'progress.pending.json';t.write_text(json.dumps(dict(status=s,PID=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=completed,**kw),indent=2)+'\n');t.replace(a.output/'progress.json')
 try:
  # Isolated gguf Python package only; parent vLLM/torch installation remains read only.
  target=a.output/'python-dependencies';target.mkdir()
  with (a.output/'dependency-install.log').open('x') as f:subprocess.run([str(a.python),'-m','pip','install','--target',str(target),'--no-deps','gguf==0.17.1'],stdout=f,stderr=subprocess.STDOUT,check=True,timeout=300)
  env=dict(os.environ,PYTHONPATH=str(target)+os.pathsep+os.environ.get('PYTHONPATH',''),OMP_NUM_THREADS='2')
  directory=a.output/'public-GGUF';directory.mkdir()
  for model,repo,rev,filename,expected in MODELS:
   state('downloading-pinned-public-GGUF',model=model,repo=repo,revision=rev)
   gguf=directory/filename;pending=gguf.with_name(gguf.name+'.partial');url=f'https://huggingface.co/{repo}/resolve/{rev}/{filename}'
   with urllib.request.urlopen(url,timeout=120) as src,pending.open('xb') as dst:
    for block in iter(lambda:src.read(16*1024*1024),b''):dst.write(block)
   assert sha(pending)==expected;pending.rename(gguf)
   audit=dict(public_repo=repo,revision=rev,filename=filename,SHA256=expected,bytes=gguf.stat().st_size,unchanged_GGUF=True)
   (directory/(model+'-download-audit.json')).write_text(json.dumps(audit,indent=2)+'\n')
   state('running-original-Marlin-same-GGUF-probe',model=model)
   argv=[str(a.python),'-B','-u',str(Path(__file__).resolve().parent/'benchmark_original_marlin.py'),'--model',model,'--et-source',str(a.source),'--gguf',str(gguf),'--output',str(a.output/(model+'-benchmark')),'--lock',str(a.lock),'--max-seconds','7200']
   (a.output/(model+'-command.json')).write_text(json.dumps(dict(argv=argv),indent=2)+'\n')
   with (a.output/(model+'-benchmark.log')).open('x') as f:rc=subprocess.run(argv,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=7500).returncode
   completed.append(dict(model=model,returncode=rc));state('model-probe-terminal',model=model,returncode=rc)
  state('campaign-terminal-review-required',ET_integrated=False)
 except Exception as e:(a.output/'exception.txt').write_text(traceback.format_exc());state('failed-preserve-evidence',detail=str(e)[-1500:]);raise
if __name__=='__main__':main()
