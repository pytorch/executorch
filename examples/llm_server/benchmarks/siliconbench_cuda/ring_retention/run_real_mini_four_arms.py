"""Real mini fixed-input numerics, latency, and lifecycle memory factorial."""
import argparse, datetime, fcntl, hashlib, json, os, selectors
import statistics, subprocess, time, traceback
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while data := f.read(16*2**20): h.update(data)
    return h.hexdigest()


class Probe:
    def __init__(self, binary, artifact, directory, trim, memory_only=False):
        directory.mkdir(parents=True)
        self.directory = directory
        self.log = (directory/'stderr.log').open('x')
        self.raw = (directory/'protocol.jsonl').open('x')
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES':'0'}
        env.pop('ET_CUDA_BENCH_MEMORY_TRACE', None)
        if memory_only:env['ET_CUDA_BENCH_MEMORY_TRACE']='1'
        env.pop('ET_CUDA_IDLE_SPARE_BYTES', None)
        if trim: env['ET_CUDA_IDLE_SPARE_BYTES'] = str(256*2**20)
        argv = [str(binary), '--pte='+str(artifact/'model.pte'),
                '--data_path='+str(artifact/'aoti_cuda_blob.ptd'),
                '--output_dir='+str(directory), '--max_sessions=21',
                '--max_context=131072', '--initial_kv_capacity=512']
        (directory/'command.json').write_text(json.dumps(dict(argv=argv,
            allocator_idle_spare_bytes=256*2**20 if trim else None), indent=2)+'\n')
        self.child = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.log, text=True, bufsize=1, env=env)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.child.stdout, selectors.EVENT_READ)
        self.serial = 0
        self.ready = self.receive()
        assert self.ready['ready'] and self.ready['snapshot']['pid']==self.child.pid

    def receive(self):
        deadline = time.monotonic()+1800
        while True:
            assert self.selector.select(max(0, deadline-time.monotonic())), 'probe reply timeout'
            line = self.child.stdout.readline()
            assert line, 'probe exited '+str(self.child.poll())
            try: reply = json.loads(line)
            except ValueError:
                self.log.write(line);self.log.flush();continue
            self.raw.write(json.dumps(dict(response=reply))+'\n');self.raw.flush()
            assert not reply.get('failed'), reply
            # RSS includes host undo pages; this is sampled outside native timings.
            status = (Path('/proc')/str(self.child.pid)/'status').read_text()
            reply['host_memory'] = {n:int(next(x.split()[1] for x in status.splitlines()
                if x.startswith(n+':')))*1024 for n in ('VmRSS','VmHWM')}
            return reply

    def request(self, body):
        self.serial += 1
        body = {**body,'tag':'request-'+str(self.serial)}
        self.raw.write(json.dumps(dict(request=body))+'\n');self.raw.flush()
        self.child.stdin.write(json.dumps(body)+'\n');self.child.stdin.flush()
        reply = self.receive();assert reply['op']==body['op'];return reply

    def step(self, ids, tokens, save=False):
        # One selected logit row per contiguous query segment, matching server.
        select = [i for i in range(len(ids)) if i==len(ids)-1 or ids[i]!=ids[i+1]]
        return self.request(dict(op='step',seq_ids=ids,tokens=tokens,
            select=select,argmax=True,save_logits=save))

    def close(self):
        try:
            if self.child.poll() is None:
                self.child.stdin.close();assert self.child.wait(timeout=120)==0
        finally:
            if self.child.poll() is None:
                self.child.terminate()
                try:self.child.wait(timeout=30)
                except subprocess.TimeoutExpired:self.child.kill();self.child.wait(timeout=30)
            self.selector.close();self.log.close();self.raw.close()


def token(owner, pos): return 100+(owner*101+pos*37)%149000


def logits(reply):
    import numpy as np
    return np.concatenate([np.fromfile(p['file'],dtype=np.float32).reshape(p['shape'])
        for p in reply['logits']],axis=0)


def compare(a,b):
    import numpy as np
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    delta=(a-b).astype(np.float64)
    denom=float(np.sqrt(np.mean(a.astype(np.float64)**2)))
    return dict(normalized_RMS=float(np.sqrt(np.mean(delta**2)))/max(denom,1e-12),
        max_abs=float(np.max(np.abs(delta))),
        argmax_matches=int(np.sum(a.argmax(-1)==b.argmax(-1))),rows=a.shape[0])


def exercise(probe, budget, window, number, memory_only=False):
    cases=[('decode'+str(c),[1]*c) for c in (1,2,4,8,16)]
    cases += [('prefill64',[64]),('prefill512',[512]),('prefill-default',[budget]),
              ('decode8-prefill64',[1]*8+[64]),('decode8-prefill512',[1]*8+[512])]
    records=[];outputs={}
    if memory_only:cases=[]
    for label,widths in cases:
        owners=[probe.request(dict(op='new'))['seq_id'] for _ in widths]
        positions=[0]*len(owners)
        # Seed exactly 1K history per owner, with matching calls in all arms.
        for i,owner in enumerate(owners):
            probe.step([owner]*1024,[token(i,p) for p in range(1024)])
        positions=[1024]*len(owners)
        before=probe.request(dict(op='metrics'))['snapshot'];calls=[]
        for cycle in range(16):
            ids=[];tokens=[]
            for i,(owner,width) in enumerate(zip(owners,widths)):
                ids += [owner]*width
                tokens += [token(i,p) for p in range(positions[i],positions[i]+width)]
                positions[i]+=width
            reply=probe.step(ids,tokens,number==1 and cycle==15)
            current=reply['snapshot'];previous={r['method']:r['calls'] for r in before['method_calls']}
            delta={r['method']:r['calls']-previous.get(r['method'],0)
                for r in current['method_calls'] if r['calls']!=previous.get(r['method'],0)}
            method='forward_'+str(next((m for m in (1,2,4,8,16,32,64) if m>=len(ids)),'others'))
            assert delta=={method:1},(label,delta)
            changed=any(current[k]!=before[k] for k in
                ('live_KV_bytes','row_growth_count','column_resize_count'))
            calls.append(dict(cycle=cycle,native_seconds=reply['elapsed_seconds'],
                storage_changed=changed,snapshot=current,host_memory=reply['host_memory'],method=method))
            before=current
            if number==1 and cycle==15:outputs[label]=reply
        stable=[c['native_seconds'] for c in calls[4:] if not c['storage_changed']]
        assert stable,(label,'no stable calls')
        records.append(dict(case=label,widths=widths,calls=calls,
            stable_mean_ms=1000*statistics.mean(stable),
            measured_mean_ms=1000*statistics.mean(c['native_seconds'] for c in calls[4:])))
        for owner in owners:probe.request(dict(op='remove',seq_id=owner))
    # Independent owners, maximum rewind, prefix clones and repeated owner runs.
    a=probe.request(dict(op='new'))['seq_id'];b=probe.request(dict(op='new'))['seq_id']
    length=0
    def fill(target):
        nonlocal length
        while length<target:
            n=min(budget,target-length)
            probe.step([a]*n,[token(0,p) for p in range(length,length+n)])
            length+=n
    fill(window+budget+17)
    start=length
    original=probe.step([a],[token(0,start)],True)
    probe.request(dict(op='rewind',seq_id=a,position=start))
    replay=probe.step([a],[token(0,start)],True)
    rewind_compare=compare(logits(original),logits(replay))
    assert rewind_compare['normalized_RMS']<1e-5,rewind_compare
    probe.request(dict(op='rewind',seq_id=a,position=start+1-budget))
    probe.request(dict(op='rewind',seq_id=a,position=start-budget,expect_success=False))
    # Clone at a retained checkpoint, then move the source independently.
    keep=start+1-budget
    clone=probe.request(dict(op='clone',seq_id=a,upto=keep))['seq_id']
    source_out=probe.step([a],[token(0,keep)],True)
    clone_out=probe.step([clone],[token(0,keep)],True)
    clone_compare=compare(logits(source_out),logits(clone_out))
    assert clone_compare['normalized_RMS']<1e-5,clone_compare
    repeat=probe.step([a,a,b,a],[token(0,keep+1),token(0,keep+2),token(1,0),token(0,keep+3)],number==1)
    if number==1:outputs['repeated-owner-after-wrap']=repeat
    for owner in (a,b,clone):probe.request(dict(op='remove',seq_id=owner))
    # Heterogeneous history: one long owner must not stretch short owners' full KV.
    owners=[];lengths={};waves=[]
    for c in (1,2,4,8,16,1):
        if c==1 and len(owners)==16:
            survivor=owners[5]
            for owner in owners:
                if owner!=survivor:probe.request(dict(op='remove',seq_id=owner))
            owners=[survivor]
        while len(owners)<c:
            owner=probe.request(dict(op='new'))['seq_id'];owners.append(owner);lengths[owner]=0
            target=12289 if len(owners)==1 else 511+(len(owners)-1)*67
            while lengths[owner]<target:
                n=min(budget,target-lengths[owner]);pos=lengths[owner]
                probe.step([owner]*n,[token(owner,p) for p in range(pos,pos+n)])
                lengths[owner]+=n
        for _ in range(8):
            probe.step(owners,[token(owner,lengths[owner]) for owner in owners])
            for owner in owners:lengths[owner]+=1
        if memory_only:time.sleep(.35)
        reply=probe.request(dict(op='metrics'))
        samples=[]
        if memory_only:
            for line in (probe.directory/'stderr.log').read_text(errors='replace').splitlines():
                if line.startswith('KV_MEM_SAMPLE:'):
                    samples.append(json.loads(line[len('KV_MEM_SAMPLE:'):]))
            assert samples, 'No native allocator diagnostic samples'
        assert reply['snapshot']['physical_columns']==c+1,reply
        waves.append(dict(concurrency=c,session_lengths={str(x):lengths[x] for x in owners},
            snapshot=reply['snapshot'],host_memory=reply['host_memory'],
            allocator_sample=samples[-1] if samples else None))
    assert waves[-1]['snapshot']['live_KV_bytes']<waves[-2]['snapshot']['live_KV_bytes']
    for owner in owners:probe.request(dict(op='remove',seq_id=owner))
    idle=probe.request(dict(op='metrics'))
    return dict(cases=records,numerics=dict(rewind=rewind_compare,clone=clone_compare),
                memory_waves=waves,idle=idle),outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for n in ('native-preparation','export-job','output','lock'):parser.add_argument('--'+n,type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists();args.output.mkdir(parents=True)
    def state(status,**extra):
        (args.output/'progress.json').write_text(json.dumps(dict(status=status,PID=os.getpid(),
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_models_paused=True,
            mini_candidate_accepted=False,**extra),indent=2)+'\n')
    try:
        gate=json.loads((args.native_preparation/'native-gate.json').read_text())
        assert gate['native_tests_passed'] and digest(gate['probe'])==gate['probe_SHA256']
        exported=json.loads((args.export_job/'result.json').read_text())
        audits={r['arm']:r for r in exported['arms']};budget=audits['baseline']['prefill_budget']
        assert audits['ring']['prefill_budget']==budget
        assert audits['ring']['GGUF_SHA256']==audits['baseline']['GGUF_SHA256']
        for audit in audits.values():
            for n,h in audit['artifacts'].items():assert digest(Path(audit['artifact'])/n)==h['SHA256']
        arms=[('baseline','baseline',False),('ring','ring',False),
              ('allocator','baseline',True),('both','ring',True)]
        records=[];saved={};memory_records=[]
        window=2048 if exported['model']=='MG' else 1024
        state('waiting-GPU-lock')
        with args.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            for number in range(1,5):
                order=arms if number%2 else list(reversed(arms))
                for arm,artifact_arm,trim in order:
                    directory=args.output/('round-'+str(number))/arm
                    state('running-real-mini-four-arms',round=number,arm=arm)
                    probe=Probe(Path(gate['probe']),Path(audits[artifact_arm]['artifact']),directory,trim)
                    try:result,outputs=exercise(probe,budget,window,number)
                    finally:probe.close()
                    records.append(dict(round=number,arm=arm,**result))
                    if number==1:saved[arm]=outputs
                    (args.output/'results.partial.json').write_text(json.dumps(records,indent=2)+'\n')
                    # The allocator sampler runs only in this separate memory
                    # diagnostic, never in measured model latency rounds.
                    state('running-real-mini-memory-only',round=number,arm=arm)
                    memory_probe=Probe(Path(gate['probe']),Path(audits[artifact_arm]['artifact']),
                        args.output/('memory-round-'+str(number))/arm,trim,True)
                    try:memory,_=exercise(memory_probe,budget,window,number,True)
                    finally:memory_probe.close()
                    memory_records.append(dict(round=number,arm=arm,**memory))
                    (args.output/'memory.partial.json').write_text(json.dumps(memory_records,indent=2)+'\n')
        numerical=[]
        for arm in ('ring','allocator','both'):
            assert saved[arm].keys()==saved['baseline'].keys()
            for label in saved['baseline']:
                item=compare(logits(saved['baseline'][label]),logits(saved[arm][label]))
                numerical.append(dict(arm=arm,case=label,**item))
        aggregates=[]
        for arm,_,_ in arms:
            retained=[r for r in records if r['arm']==arm and r['round']>1]
            for i,case in enumerate(retained[0]['cases']):
                values=[r['cases'][i]['stable_mean_ms'] for r in retained]
                aggregates.append(dict(arm=arm,case=case['case'],mean_ms=statistics.mean(values),
                    sample_SD_ms=statistics.stdev(values),rounds=values))
        result=dict(model=exported['model'],records=records,memory_records=memory_records,latency=aggregates,
            cross_arm_numerics=numerical,four_rounds_discard_first=True,
            native_gate_SHA256=digest(args.native_preparation/'native-gate.json'),
            numerical_and_performance_review_required=True,mini_candidate_accepted=False)
        (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        state('real-mini-four-arm-data-complete-review-pending')
    except BaseException as e:
        state('failed-real-mini-four-arm',error=repr(e),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
