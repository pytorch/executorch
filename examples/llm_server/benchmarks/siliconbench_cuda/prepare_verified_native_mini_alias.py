"""Reuse exact mini artifacts to isolate an already tested native SDK stage.

Create aliases and verified provenance only. Do not rebuild, change source or
models, replace prior runs, or qualify a full model.
"""
import argparse,datetime,json,os
from pathlib import Path
from independent_session_full_native_binding import alias_runtime,sha,verify_runtime
from independent_session_mini_profiles import PROFILES,validate_mini_audit


def main():
 p=argparse.ArgumentParser(description=__doc__)
 for n in ('runtime-job','original-mini-preparation','output'):
  p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--profile',choices=PROFILES,required=True)
 a=p.parse_args();package=Path(__file__).resolve().parent
 gate=verify_runtime(a.runtime_job,package)
 old=json.loads((a.original_mini_preparation/'progress.json').read_text())
 assert old['status']=='mini-artifacts-ready'
 original_gate=json.loads((a.original_mini_preparation/'native-gate.json').read_text())
 assert gate['source_spec_SHA256']==original_gate['source_spec_SHA256']
 assert gate['unchanged_controls']==original_gate['unchanged_controls']
 artifacts={}
 for arm in ('global','independent'):
  artifact=a.original_mini_preparation/f'mini-{arm}-artifact'
  audit_path=artifact/'independent-mini-artifact-audit.json';audit=json.loads(audit_path.read_text())
  validate_mini_audit(audit,a.profile,arm=='independent')
  assert audit['source_spec_sha256']==gate['source_spec_SHA256']
  for name,r in audit['artifacts'].items():assert sha(artifact/name)==r['sha256'],name
  artifacts[arm]={'artifact':str(artifact.resolve()),'audit_SHA256':sha(audit_path),'artifacts':audit['artifacts']}
 assert not a.output.exists();a.output.mkdir(parents=True)
 alias_runtime(a.runtime_job,package,a.output,original_gate['source'])
 for arm,item in artifacts.items():(a.output/f'mini-{arm}-artifact').symlink_to(item['artifact'])
 (a.output/'proof.json').write_text(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  runtime_native_gate_SHA256=sha(a.runtime_job/'native-gate.json'),
  original_mini_gate_SHA256=sha(a.original_mini_preparation/'native-gate.json'),
  model_artifacts=artifacts,native_tests74_reverified=True,source_kernel_and_ABI_controls_reverified=True,
  models_and_sources_unchanged=True,diagnostic_stage_only=True,full_model_qualified=False,
  main_candidate_accepted=False),indent=2)+'\n')
 (a.output/'progress.json').write_text(json.dumps(dict(status='mini-artifacts-ready',
  controller_pid=os.getpid(),original_compiled_artifacts_unchanged=True,
  diagnostic_native_stage=True,full_model_qualified=False,main_candidate_accepted=False),indent=2)+'\n')
 print(json.dumps({'status':'mini-artifacts-ready','diagnostic_native_stage':True}))

if __name__=='__main__':main()
