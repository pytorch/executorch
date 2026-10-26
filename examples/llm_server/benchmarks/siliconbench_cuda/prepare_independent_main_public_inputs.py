"""Prepare pinned public task fixtures in a separate CPU-only benchmark venv.

No GPU use or framework environment change. Dataset and rendered0/5-shot prompt
hashes must equal the existing main comparison. No source data is published.
"""
import argparse, csv, datetime, hashlib, json, os, shutil, subprocess, sys, urllib.request
from collections import Counter
from pathlib import Path

DATA_REVISION='186fd01f69c9036aed5a723883fb5537dd2ed821'
DATA_FILES={
 'GMRID_v3-test.csv':('9c7ab1f78af00bdab209646cb5f36c95cf63635035f607b0a3e9f7894e0a4d18',2301669),
 'categories.json':('82c0c1f7995b803a8c5ec9ff78fba8c5da3ad6f36fb8ee3c205e22794863cb0a',3053)}
PROMPT_HASHES={
 '0':'1ad52f828b58473063e7a4c0265bd9cabbbea6c08ed9898ee8109563a61d8d73',
 '5':'94b72cc81a0dc11f32c1fdf07873e40e2ecc6443c345dd6cd8c002b3a1ad924c'}
LABEL_SHA='b2bd39c11cd52f15a2b87d64cd0e61d92eb958a737f8edc0cece204c73753b2b'
BENCH_REVISION='3e9fa20c6fc6de333c5cb40e5da46d001194443c'


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
 a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
 package=Path(__file__).resolve().parent
 def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
 def state(status,**extra):
  temporary=a.output/'progress.pending.json'
  temporary.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),
   UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU_work_started=False,framework_environments_modified=False,**extra),indent=2)+'\n')
  temporary.replace(a.output/'progress.json')
 def fetch(url,path,expected=None,count=None):
  with urllib.request.urlopen(url,timeout=120) as incoming:raw=incoming.read()
  value=hashlib.sha256(raw).hexdigest()
  assert expected is None or value==expected,url
  assert count is None or len(raw)==count,url
  path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
  return dict(url=url,sha256=value,bytes=len(raw))
 try:
  state('preparing-separate-CPU-benchmark-venv')
  uv=shutil.which('uv');assert uv
  python=a.output/'.venv/bin/python'
  with (a.output/'dependencies.log').open('x') as log:
   subprocess.run([uv,'venv','--python','3.12',str(a.output/'.venv')],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600)
   subprocess.run([uv,'pip','install','--python',str(python),'aiohttp==3.13.3','scikit-learn==1.7.2'],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=600)
  data=a.output/'correctness/data';proof=dict(source_repository='inflaton/llms-at-edge',revision=DATA_REVISION,files={})
  state('downloading-pinned-public-correctness-data')
  for name,(expected,count) in DATA_FILES.items():
   proof['files'][name]=fetch(f'https://raw.githubusercontent.com/inflaton/llms-at-edge/{DATA_REVISION}/dataset/{name}',data/name,expected,count)
  with (data/'GMRID_v3-test.csv').open(newline='') as stream:records=list(csv.DictReader(stream))
  categories=json.loads((data/'categories.json').read_text());assert len(records)==1147 and len(categories)==8
  assert len({str(row['id']) for row in records})==1147
  valid=[];excluded=[]
  for row in records:
   reasons=[]
   if not row['Headline_Details'].strip():reasons.append('empty_input')
   if row['Summarized_label'].strip() not in categories:reasons.append('unknown_gold_label')
   if reasons:excluded.append(dict(id=str(row['id']),reasons=reasons,gold_label=row['Summarized_label'].strip()))
   else:valid.append(row)
  assert len(valid)==1146 and excluded==[dict(id='5079',reasons=['empty_input'],gold_label='Others')]
  scripts=package/'correctness/scripts';(a.output/'correctness/scripts').symlink_to(scripts)
  proof.update(source_rows=1147,evaluated_items=1146,excluded_source_rows=excluded,
   gold_label_distribution=dict(Counter(row['Summarized_label'].strip() for row in valid)),prompt_sets={})
  for shots in (0,5):
   target=a.output/f'correctness/prompts/{shots}shot'
   with (a.output/f'build-prompts-{shots}.log').open('x') as log:
    subprocess.run([str(python),'-B',str(scripts/'build_prompts.py'),'--shots',str(shots),
     '--csv',str(data/'GMRID_v3-test.csv'),'--categories',str(data/'categories.json'),
     '--output-dir',str(target)],check=True,stdout=log,stderr=subprocess.STDOUT,timeout=120)
   assert sha(target/'prompts.jsonl')==PROMPT_HASHES[str(shots)] and sha(target/'labels.jsonl')==LABEL_SHA
   proof['prompt_sets'][str(shots)]=dict(prompts_sha256=sha(target/'prompts.jsonl'),labels_sha256=LABEL_SHA,
    count=1146,builder_no_think_default=True,max_tokens=256)
  proof['scripts']={path.name:sha(path) for path in scripts.iterdir() if path.suffix=='.py'}
  (a.output/'GMRID-input-audit.json').write_text(json.dumps(proof,indent=2)+'\n')
  benchmark={}
  for split in ('chat','agent'):
   benchmark[split]=fetch(f'https://huggingface.co/datasets/windchimeran/SiliconBench/resolve/{BENCH_REVISION}/{split}_benchmark_prompts.json',a.output/f'{split}_benchmark_prompts.json')
   assert json.loads((a.output/f'{split}_benchmark_prompts.json').read_text())
  (a.output/'SiliconBench-input-audit.json').write_text(json.dumps(dict(revision=BENCH_REVISION,splits=benchmark,
   correctness_original_main_hashes_equal=True,benchmark_python=str(python),framework_environments_modified=False),indent=2)+'\n')
  state('pinned-public-main-inputs-ready',correctness_items=1146,correctness_shots=[0,5],splits=['chat','agent'])
 except Exception as error:
  state('failed',error=str(error));raise


if __name__=='__main__':main()
