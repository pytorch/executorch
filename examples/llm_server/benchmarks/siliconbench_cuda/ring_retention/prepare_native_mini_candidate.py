"""Build the isolated native mini candidate and run old/new storage gates."""
import argparse,datetime,fcntl,hashlib,json,os,shutil,subprocess,sys,traceback
from pathlib import Path
import xml.etree.ElementTree as ET

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 a=argparse.ArgumentParser(description=__doc__)
 for n in ['baseline-native','baseline-export','output','python','lock']:a.add_argument('--'+n,type=Path,required=True)
 a.add_argument('--jobs',type=int,default=4);a.add_argument('--recover-terminal-job',type=Path);args=a.parse_args()
 assert not args.output.exists();args.output.mkdir(parents=True)
 package=Path(__file__).parent;prepared=args.recover_terminal_job or args.output
 source=prepared/'native/executorch';export=prepared/'export-source/executorch'
 def state(status,**extra):
  d=dict(status=status,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),PID=os.getpid(),source=str(source),export_source=str(export),mini_model_qualified=False,original_campaign_paused=True,**extra)
  p=args.output/'progress.pending.json';p.write_text(json.dumps(d,indent=2)+'\n');p.replace(args.output/'progress.json')
 env={**os.environ,'CUDA_VISIBLE_DEVICES':'','CUDA_HOME':'/usr/local/cuda','CUDAARCHS':'120','TORCH_CUDA_ARCH_LIST':'12.0','PYTHONDONTWRITEBYTECODE':'1','MAX_JOBS':str(args.jobs),'OMP_NUM_THREADS':str(args.jobs)}
 env['PATH']=str(args.python.parent)+os.pathsep+env['PATH'];env['PYTHONPATH']=str(source.parent)
 def run(argv,log='build.log',gpu=False,extra=None,timeout=14400):
  e={**env,**(extra or {})};e['CUDA_VISIBLE_DEVICES']='0' if gpu else ''
  with (args.output/log).open('a') as f:
   f.write(json.dumps([str(x) for x in argv])+'\n');f.flush()
   subprocess.run([str(x) for x in argv],stdout=f,stderr=subprocess.STDOUT,env=e,check=True,timeout=timeout)
 try:
  state('copying-and-patching-isolated-native-export-sources')
  if args.recover_terminal_job:
   prior=json.loads((prepared/'progress.json').read_text())
   assert prior['status']=='failed-native-mini-candidate' and not (Path('/proc')/str(prior['PID'])).exists()
   audit=json.loads((source/'ring-retention-source-audit.json').read_text())
   for n,h in audit['candidate_SHA256'].items():assert sha(source/n)==h,n
   cpp=source/'backends/cuda/runtime/cuda_kv_cache.cpp';before=cpp.read_text()
   target='        if (cfg_.split_ring_history && !copy_current_ring) continue;'
   assert before.count(target)==1
   # The old failed candidate accidentally placed this guard in a separate
   # sequence factory. Preserve its exact source before fixing its scope.
   (args.output/'failed-candidate-before-scope-fix.cpp').write_text(before)
   after=before.replace(target+'\n','')
   target='      const cache::LayerGeometry& layer = geometry_.layers[index];\n      const size_t slot'
   assert after.count(target)==1
   after=after.replace(target,'      const cache::LayerGeometry& layer = geometry_.layers[index];\n      if (cfg_.split_ring_history && is_ring(layer) && !copy_current_ring) continue;\n      const size_t slot')
   cpp.write_text(after)
   (args.output/'source-recovery.json').write_text(json.dumps(dict(prior_job=str(prepared),prior_progress_SHA256=sha(prepared/'progress.json'),before_file=str(args.output/'failed-candidate-before-scope-fix.cpp'),before_SHA256=hashlib.sha256(before.encode()).hexdigest(),after_SHA256=sha(cpp),prior_build_log_preserved=True,only_failed_owned_source_changed=True),indent=2)+'\n')
  else:
   run([args.python,package/'apply_ring_retention_candidate.py','--baseline',args.baseline_native,'--output',source,'--role','native'])
   run([args.python,package/'apply_ring_retention_candidate.py','--baseline',args.baseline_export,'--output',export,'--role','export'])
  cmake=shutil.which('cmake',path=env['PATH']);assert cmake
  prefix=subprocess.check_output([args.python,'-B','-c','import torch; print(torch.utils.cmake_prefix_path)'],text=True,env=env).strip()
  build=source/'cmake-out';state('building-isolated-native-core')
  run([cmake,'-S',source,'-B',build,'-DCMAKE_BUILD_TYPE=Release','-DEXECUTORCH_BUILD_PRESET_FILE='+str(source/'tools/cmake/preset/llm.cmake'),'-DEXECUTORCH_BUILD_CUDA=ON','-DEXECUTORCH_BUILD_PYBIND=OFF','-DEXECUTORCH_BUILD_TESTS=ON','-DBUILD_TESTING=ON','-DCMAKE_INSTALL_PREFIX='+str(build),'-DCMAKE_PREFIX_PATH='+prefix,'-DPYTHON_EXECUTABLE='+str(args.python)])
  run([cmake,'--build',build,'--parallel',str(args.jobs),'--target','test_cuda_kv_cache'])
  test=build/'backends/cuda/test_cuda_kv_cache';state('waiting-storage-GPU-gates')
  with args.lock.open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX)
   assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
   xml=args.output/'native-baseline-policy.xml';state('running-native-storage-gates')
   run([test,'--gtest_output=xml:'+str(xml)],log='native-gates.log',gpu=True,timeout=1800)
   tests=list(ET.parse(xml).getroot().iter('testcase'))
   assert len(tests)>=78 and all(t.find('failure') is None and t.find('error') is None and t.find('skipped') is None for t in tests)
   names=[t.attrib['name'] for t in tests if t.attrib['classname']=='CudaSplitRingCacheTest'];assert len(names)==4,names
   xml2=args.output/'native-allocator-policy.xml'
   run([test,'--gtest_filter=CudaSplitRingCacheTest.*','--gtest_output=xml:'+str(xml2)],log='native-allocator-gates.log',gpu=True,extra={'ET_CUDA_IDLE_SPARE_BYTES':str(256*2**20)},timeout=1800)
   policy=list(ET.parse(xml2).getroot().iter('testcase'));assert len(policy)==4 and all(t.find('failure') is None and t.find('error') is None for t in policy)
  state('installing-isolated-native-SDK',storage_gate_count=len(tests))
  run([cmake,'-S',source,'-B',build,'-DEXECUTORCH_BUILD_TESTS=OFF','-DBUILD_TESTING=OFF'])
  run([cmake,'--build',build,'--parallel',str(args.jobs),'--target','install'])
  # Reuse the existing diagnostic protocol, adding rewind and the unchanged
  # optional sampler. No prior qualification receipt is relabelled as new.
  official=source/'examples/models/muse-glimmer/tests/runtime_kv_columns_probe.cpp';text=official.read_text()
  assert text.count('#include <unistd.h>')==1;text=text.replace('#include <unistd.h>','#include "kv_memory_sampler.h"\n#include <unistd.h>')
  anchor='    require(executor->initialize(), "initialize actual CUDA executor");';assert text.count(anchor)==1
  text=text.replace(anchor,anchor+'\n    siliconbench::KVMemorySampler sampler(*executor, "cuda", false);')
  anchor='      } else if (op == "compact") {';assert text.count(anchor)==1
  text=text.replace(anchor,'      } else if (op == "rewind") {\n        const auto sid=request.at("seq_id").get<int32_t>();\n        const bool rewound=control.rewind(sid,request.at("position").get<int>());\n        require(rewound == request.value("expect_success",true), "rewind expectation");\n        result["rewound"]=rewound;result["position"]=control.pos(sid);\n'+anchor)
  cpp=args.output/'mini_probe.cpp';cpp.write_text(text)
  pb=args.output/'probe-build';parent=package.parent
  run([cmake,'-S',parent/'rewind_kv_probe_project','-B',pb,'-DCMAKE_BUILD_TYPE=Release','-DET_SOURCE='+str(source),'-DREWIND_PROBE_SOURCE='+str(cpp),'-DCMAKE_PREFIX_PATH='+str(build)+';'+prefix])
  run([cmake,'--build',pb,'--parallel',str(args.jobs),'--target','rewind_kv_probe'])
  binary=pb/'rewind_kv_probe'
  receipt=dict(native_tests_passed=True,legacy_test_count=len(tests)-4,new_ring_test_names=names,allocator_policy_test_count=len(policy),source=str(source),export_source=str(export),probe=str(binary),probe_SHA256=sha(binary),native_baseline_policy_XML_SHA256=sha(xml),allocator_policy_XML_SHA256=sha(xml2),native_source_audit_SHA256=sha(source/'ring-retention-source-audit.json'),export_source_audit_SHA256=sha(export/'ring-retention-source-audit.json'),kernel_qualification_pending_until_model_export=True,real_mini_model_qualified=False,actual_native_Cpp_SHA256=sha(source/'backends/cuda/runtime/cuda_kv_cache.cpp'),source_recovery_SHA256=sha(args.output/'source-recovery.json') if args.recover_terminal_job else None)
  (args.output/'native-gate.json').write_text(json.dumps(receipt,indent=2)+'\n');state('native-mini-candidate-ready',native_gate=str(args.output/'native-gate.json'),probe=str(binary))
 except BaseException as e:
  state('failed-native-mini-candidate',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
