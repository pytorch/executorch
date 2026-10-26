"""Install pinned HTTP control-plane dependencies into a new isolated target.

The model/export Python environment, native SDK and serving sources are kept.
Only public PyPI wheels are fetched. No model is loaded by the import check.
"""
import argparse, datetime, hashlib, json, os, shutil, subprocess, traceback
from pathlib import Path

VERSIONS={'fastapi':'0.159.1','uvicorn':'0.54.0','pydantic':'2.14.1',
          'pydantic_core':'2.50.1','starlette':'1.9.0','jinja2':'3.1.6',
          'anyio':'4.15.1','click':'8.5.0','h11':'0.16.0',
          'typing_extensions':'4.16.0','annotated-types':'0.8.0','MarkupSafe':'3.0.4'}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('python','native-job','output'):
  p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 def state(status,**extra):
  pending=a.output/'progress.pending.json'
  pending.write_text(json.dumps({'status':status,'controller_pid':os.getpid(),
    'UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'CPU_only':True,
    'candidate_accepted':False,**extra},indent=2)+'\n')
  pending.replace(a.output/'progress.json')
 try:
  package=Path(__file__).resolve().parent
  from independent_session_full_native_binding import verify_runtime,sha
  gate=verify_runtime(a.native_job,package)
  source=Path(gate['source'])
  serve=source/'examples/models/muse-glimmer/serving/serve_batching.py'
  target=a.output/'python-dependencies'
  env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'}
  check='import importlib.metadata as m,json;print(json.dumps({k:m.version(k) for k in ("torch","triton")}))'
  before=json.loads(subprocess.check_output([str(a.python),'-B','-c',check],env=env,text=True))
  has_pip=subprocess.check_output([str(a.python),'-B','-c',
      'import importlib.util;print(int(importlib.util.find_spec("pip") is not None))'],env=env,text=True).strip()=='1'
  if has_pip:
   installer=[str(a.python),'-B','-m','pip','install','--disable-pip-version-check','--only-binary=:all:']
  else:
   uv=shutil.which('uv');assert uv,'Neither pip nor the existing uv installer is available'
   installer=[uv,'pip','install','--python',str(a.python),'--only-binary',':all:']
  state('installing-isolated-public-HTTP-dependencies')
  with (a.output/'install.log').open('x') as log:
   subprocess.run([*installer,'--target',str(target),*[k+'=='+v for k,v in VERSIONS.items()]],
      check=True,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=600)
  env['PYTHONPATH']=str(target)+os.pathsep+str(source.parent)
  code='''import importlib.metadata as m,importlib.util,json,sys
versions={k:m.version(k) for k in EXPECTED}
assert versions==EXPECTED,versions
spec=importlib.util.spec_from_file_location('http_import_check',SERVE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
assert callable(module.main)
print(json.dumps({'versions':versions,'official_server_import_pass':True}))
'''
  check_result=json.loads(subprocess.check_output([str(a.python),'-B','-c',
      'EXPECTED='+repr(VERSIONS)+'\nSERVE='+repr(str(serve))+'\n'+code],
      env=env,text=True,timeout=180))
  after=json.loads(subprocess.check_output([str(a.python),'-B','-c',check],
      env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},text=True))
  assert before==after
  assert verify_runtime(a.native_job,package)==gate
  files={str(path.relative_to(target)):hashlib.sha256(path.read_bytes()).hexdigest()
         for path in sorted(target.rglob('*')) if path.is_file()}
  proof={'dependency_path':str(target),'versions':VERSIONS,'files_SHA256':files,
         'native_gate_SHA256':sha(a.native_job/'native-gate.json'),
         'base_torch_triton_versions_unchanged':before==after,'base_versions':before,
         'installed_only_into_new_owned_target':True,'native_SDK_and_models_unchanged':True,
         'installer_argv':installer,
         'server_source_SHA256':sha(serve),
         **check_result}
  (a.output/'proof.json').write_text(json.dumps(proof,indent=2)+'\n')
  state('MG-HTTP-Python-dependencies-ready',proof=str(a.output/'proof.json'))
 except Exception as error:
  state('failed',detail=str(error),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
