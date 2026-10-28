"""Align proven full-KV RMS configuration differences in a fresh mini export."""
import argparse,ast,datetime,hashlib,importlib.util,json,os,re,shutil,subprocess,textwrap,traceback
from pathlib import Path

TARGETS={
 '2ffc717d03adbcac759f88d88117e1aeb02d5d1397dc7fe03cfbd9ddb8993e9c':(2048,8,4,4096),
 'de5b8197b66921a45fc1ebddc6124e4a90350eb8303149ccdb42e16c37bc2b89':(16,1,4,16),
 '41d900d84cbd2aba29cb4be7d062b1c476e9593435c1df5ab370ae0197418951':(32,1,4,16),
}
ALLOWED_ABSENT_OLD={
 'c525647649f0a2cdd6a005034e19842bdc2c2794a14a3b8fc98d72a6c20e26ed',
 'ff012d7811972473644a3bc1778da5287fa3a92d1e2399801d0f77bcfa40be6d',
}

def extract_reference(root,layout):
 rows={}
 for p in root.glob('*/*.wrapper.cpp'):
  source=p.read_text()
  for match in re.finditer(r'void call_(triton_per[^\s(]+)\([^{]*\)\{(.*?)\n\}',source,re.S):
   name,body=match.groups()
   if 'r0_numel = 512' not in body or 'rsqrt' not in body:continue
   start=body.index('async_compile.triton(');start=body.index("'''",start)+3;end=body.index("'''",start)
   embedded=textwrap.dedent(body[start:end]).replace('\\"','"');tree=ast.parse(embedded)
   fn=next(x for x in tree.body if isinstance(x,ast.FunctionDef));generated=ast.get_source_segment(embedded,fn);key=layout.fingerprint(generated)
   if key not in TARGETS:continue
   hints=None
   for d in fn.decorator_list:
    if isinstance(d,ast.Call):
     for kw in d.keywords:
      if kw.arg=='size_hints':hints=ast.literal_eval(kw.value)
   launch=re.findall(r'launchKernel\([^;]+, grid_0, grid_1, grid_2, ([0-9]+), ([0-9]+), kernel_args_, stream_\);',body);assert len(launch)==1
   xblock=re.search(r'uint32_t grid_0 = \(\(xnumel \+ \(([0-9]+) - 1\)\) / \(([0-9]+)\)\);',body)
   if xblock:assert xblock[1]==xblock[2];xb=int(xblock[1])
   else:assert 'uint32_t grid_0 = xnumel;' in body;xb=1
   actual=(hints['x'],xb,int(launch[0][0]),int(launch[0][1]));assert actual==TARGETS[key],(key,actual)
   row=dict(M=hints['x'],function_AST_SHA256=key,generated_function_source=generated,kwargs=dict(XBLOCK=xb),num_warps=actual[2],num_stages=1)
   if key in rows:assert rows[key]['kwargs']==row['kwargs'] and rows[key]['num_warps']==row['num_warps']
   rows[key]=row
 assert set(rows)==set(TARGETS),set(rows)
 return [rows[k] for k in sorted(rows)]

def main(args):
 job,template,base=args.job,args.template,args.base;assert job.is_dir() and not (job/'export_mini81.py').exists();rc=1
 def state(status,**extra):
  p=job/'progress.pending.json';p.write_text(json.dumps(dict(status=status,mini_only=True,qualified=False,PID=os.getpid(),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),**extra)));p.replace(job/'progress.json')
 try:
  assert shutil.disk_usage(base).free>=3*2**30
  for name in ['reference_generated_reduction_layout.py','reuse_exact_ptd_export.py']:(job/name).write_bytes((template/name).read_bytes())
  spec=importlib.util.spec_from_file_location('reference_layout',job/'reference_generated_reduction_layout.py');layout=importlib.util.module_from_spec(spec);spec.loader.exec_module(layout)
  reference=extract_reference(args.baseline_cache,layout);controls=json.loads((template/'Gemma-reference-reduction-layouts.json').read_text());assert len(controls)==5
  assert not set(TARGETS)&{x['function_AST_SHA256'] for x in controls};controls+=reference
  (job/'Gemma-reference-reduction-layouts.json').write_text(json.dumps(controls,indent=2));(job/'full-KV-RMS-controls-required.json').write_text(json.dumps(dict(controls=reference,prior_controls_preserved=5,baseline_actual_wrapper_configs_verified=True,unchanged_generated_functions=True,no_activation_replacement=True),indent=2))
  source=(template/'export_mini81.py').read_text().replace(str(template),str(job));ast.parse(source);(job/'export_mini81.py').write_text(source)
  argv=[a.replace(str(template),str(job)) for a in json.loads((template/'export-command.json').read_text())];cache=job/'export-cache';cache.mkdir();argv[argv.index('--cache-root')+1]=str(cache)
  (job/'export-command.json').write_text(json.dumps(argv,indent=2));(job/'data/intermediate-weights').mkdir(parents=True)
  env=dict(os.environ,ET_DIAGNOSTIC_WEIGHT_STORAGE_DIR=str(job/'data/intermediate-weights'),PYTHONPATH=str(base/'root-normal-Gemma-source-complete-qualification-20261028-v6/code'),ET_INT4_REFERENCE_EXPORT_CONFIGS=str(base/'root-authorized-ring-source-host-pool-validation-20261027-v1/controls/Gemma-baseline-INT4-config-manifest.json'))
  for n in ['ET_INT5_REFERENCE_EXPORT_CONFIGS','ET_INT6_REFERENCE_EXPORT_CONFIGS','ET_CUDA_RING_JOURNAL_BUFFERS','ET_CUDA_RING_ARCHIVE_BATCH_ROWS']:env.pop(n,None)
  state('exporting-mini-with-reference-full-KV-RMS-configs')
  with (job/'export.log').open('x') as log:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
  receipts=[json.loads(l) for l in (job/'reduction-layout-actual-control-receipt.jsonl').read_text().splitlines()];matched={x['function_AST_SHA256']:x for x in receipts}
  # Old AST controls are retained. Only the previously proven absent ASTs may be inactive.
  generated=set()
  for p in (cache/'ring/inductor').glob('*/*.py'):
   s=p.read_text();tree=ast.parse(s)
   for fn in tree.body:
    if isinstance(fn,ast.FunctionDef):generated.add(layout.fingerprint(ast.get_source_segment(s,fn)))
  absent={x['function_AST_SHA256'] for x in controls}-generated;assert absent<=ALLOWED_ABSENT_OLD,absent
  for row in controls:
   key=row['function_AST_SHA256']
   if key in absent:continue
   assert key in matched and all(matched[key][k]==row[k] for k in ['kwargs','num_warps','num_stages']),(key,matched.get(key))
  result=json.loads((job/'data/exports/result.json').read_text());assert len(result['arms'])==1
  assert result['arms'][0]['metadata']['get_offgraph_kv_max_seqs']==81 and result['arms'][0]['metadata']['get_max_prefill_chunk']==2048
  (job/'result.json').write_text(json.dumps(dict(export=result,actual_reference_config_receipts=receipts,verified_absent_old_ASTs=sorted(absent),all_applicable_controls_matched=True,numerical_qualification_passed=False,performance_qualification_passed=False,full_model_unchanged=True),indent=2));state('mini-export-complete-qualification-pending');rc=0
 except BaseException:(job/'failure.txt').write_text(traceback.format_exc());state('mini-export-failed')
 finally:(job/'exit-receipt.json').write_text(json.dumps(dict(returncode=rc,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),numerical_qualification_passed=False,full_model_unchanged=True)))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',type=Path,required=True);p.add_argument('--job',type=Path,required=True);p.add_argument('--template',type=Path,required=True);p.add_argument('--baseline-cache',type=Path,required=True);main(p.parse_args())
