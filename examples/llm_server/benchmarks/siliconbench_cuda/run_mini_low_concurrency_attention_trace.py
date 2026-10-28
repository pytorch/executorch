"""Find the first differing real attention inputs at c1/c2/c4; never timed."""
import argparse,datetime,fcntl,hashlib,json,os,subprocess,traceback
from pathlib import Path

def main(config,output):
 cfg=json.loads(config.read_text());output.mkdir(exist_ok=False);(output/'configuration.json').write_text(json.dumps(cfg,indent=2))
 for k,want in cfg['SHA256'].items():assert hashlib.sha256(Path(cfg[k]).read_bytes()).hexdigest()==want,k
 lib=Path(cfg.get('trace_library',str(output/'trace.so')));assert not lib.exists();cmd=['/usr/bin/c++','-O2','-std=c++17','-fPIC','-shared',cfg['interposer_source'],'-I'+cfg['cuda_include'],'-L'+cfg['cuda_lib'],'-Wl,-rpath,'+cfg['cuda_lib'],'-lcuda','-lcudart','-ldl','-o',str(lib)]
 (output/'build-command.json').write_text(json.dumps(cmd))
 with (output/'build.log').open('x') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 rows=[]
 with Path(cfg['lock']).open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  for c in cfg.get("concurrencies",[1,2,4]):
   for arm in ['baseline','ring']:
    folder=output/str(c)/arm;folder.mkdir(parents=True);trace=folder/'trace';trace.mkdir()
    env=dict(os.environ,LD_PRELOAD=str(lib),LD_LIBRARY_PATH=cfg['native_lib'],ET_DIAGNOSTIC_ATTENTION_OUTPUT=str(trace),ET_CUDA_RING_JOURNAL_BUFFERS='0',ET_CUDA_RING_ARCHIVE_BATCH_ROWS='64',ET_CUDA_RING_ASYNC_DMA='0')
    pte=cfg['baseline_PTE'] if arm=='baseline' else cfg['PTE'];cmd=[cfg['probe'],'--pte='+pte,'--data_path='+cfg['PTD'],'--max_sessions=81','--max_context=131072','--initial_kv_capacity=512','--output_dir='+str(folder)]
    with (folder/'stderr.log').open('x') as log:
     child=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,env=env,text=True,bufsize=1)
     def receive(req=None):
      line=child.stdout.readline();assert line,('probe ended',c,arm,child.poll());reply=json.loads(line)
      with (folder/'protocol.jsonl').open('a') as f:f.write(json.dumps(dict(request=req,response=reply))+'\n')
      assert not reply.get('failed'),reply;return reply
     def send(req):child.stdin.write(json.dumps(req)+'\n');child.stdin.flush();return receive(req)
     try:
      assert receive()['ready'];ids=[send(dict(op='new'))['seq_id'] for _ in range(c)];positions={s:0 for s in ids}
      def step(counts,tag):
       owners=[];tokens=[];select=[]
       for sid,n in counts:
        owners.extend([sid]*n);tokens.extend(100+(sid*101+i*37)%149000 for i in range(positions[sid],positions[sid]+n));positions[sid]+=n;select.append(len(tokens)-1)
       return send(dict(op='step',seq_ids=owners,tokens=tokens,select=select,argmax=False,save_logits=True,tag=tag))
      for sid in ids:step([(sid,1024)],'seed-'+str(sid))
      for i in range(66):step([(sid,1) for sid in ids],'decode-'+str(i))
      child.stdin.close();assert child.wait(timeout=120)==0
     finally:
      if child.poll() is None:child.terminate();child.wait(timeout=120)
    meta=list(trace.glob('*-metadata.json'));assert len(meta)==66*6,(c,arm,len(meta))
    rows.append(dict(c=c,arm=arm,captures=len(meta),directory=str(folder)));(output/'progress.json').write_text(json.dumps(rows))
 (output/'result.json').write_text(json.dumps(dict(cases=rows,graphs_disabled=True,unchanged_launches=True,diagnostic_only=True,full_model_qualified=False),indent=2))
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--config',type=Path,required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();rc=1
 try:main(args.config,args.output);rc=0
 except BaseException:
  if args.output.exists():(args.output/'failure.txt').write_text(traceback.format_exc())
  traceback.print_exc()
 finally:
  if args.output.exists():(args.output/'exit-receipt.json').write_text(json.dumps(dict(returncode=rc,diagnostic_only=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())))
 raise SystemExit(rc)
