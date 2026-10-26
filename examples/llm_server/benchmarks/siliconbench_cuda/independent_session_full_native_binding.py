"""Verify mini permission and separate export-source from compatible native SDK."""
import hashlib,json
from pathlib import Path
import xml.etree.ElementTree as ET

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def verify_qualification(path,profile,gguf_SHA256,runtime_job):
 q=json.loads(Path(path).read_text())
 assert q['mini_profile']==profile and q['GGUF_SHA256']==gguf_SHA256
 assert q['mini_storage_correctness_pass'] and q['mini_cold_and_steady_performance_pass'] and q['mini_HTTP_memory_pass']
 assert q['full_model_export_for_qualification_allowed'] and not q['positive_performance_signals']
 assert not q['full_model_qualified'] and not q['main_candidate_accepted']
 assert sha(Path(runtime_job)/'native-gate.json')==q['runtime_native_gate_SHA256']
 for name,expected in q['evidence_files_SHA256'].items():assert sha(name)==expected,name
 return q

def verify_runtime(runtime_job,package):
 runtime_job,package=Path(runtime_job),Path(package)
 gate=json.loads((runtime_job/'native-gate.json').read_text())
 assert gate['native_GPU_tests_pass'] and gate['test_count']==74
 assert sha(runtime_job/'native-tests.xml')==gate['native_tests_xml_SHA256']
 tests=list(ET.parse(runtime_job/'native-tests.xml').getroot().iter('testcase'))
 assert len(tests)==74 and not any(t.find('failure') is not None or t.find('error') is not None or t.find('skipped') is not None for t in tests)
 for path,expected in gate['binaries'].items():assert sha(path)==expected,path
 spec_path=package/'independent_session_capacity_source.json';spec=json.loads(spec_path.read_text())
 assert sha(spec_path)==gate['source_spec_SHA256']
 expected={name:r['after_SHA256'] for name,r in spec['files'].items()}
 changes=gate.get('actual_native_source_change')
 assert changes
 records=changes['patches'] if 'patches' in changes else [changes]
 for record in records:
  assert record['file'] in expected and record['before_SHA256']==expected[record['file']]
  expected[record['file']]=record['after_SHA256']
 for name,value in expected.items():assert sha(Path(gate['source'])/name)==value,name
 for name,value in gate['unchanged_controls'].items():assert sha(Path(gate['source'])/name)==value,name
 return gate

def alias_runtime(runtime_job,package,output,export_source,extra=None):
 gate=verify_runtime(runtime_job,package)
 output=Path(output);build=output/'model-build';build.mkdir()
 aliases=dict(gate);aliases['binaries']=dict(gate['binaries'])
 for name in ('muse_glimmer_runtime_kv_probe','muse_glimmer_batching_worker'):
  original=Path(runtime_job)/'model-build'/name;target=build/name;target.symlink_to(original)
  aliases['binaries'][str(target)]=sha(original)
 aliases['artifact_export_source']=str(export_source)
 aliases['artifact_export_and_native_SDK_sources_are_separately_verified']=True
 if extra:aliases.update(extra)
 (output/'native-gate.json').write_text(json.dumps(aliases,indent=2)+'\n')
 return aliases
