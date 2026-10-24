"""Diagnose preserved real-GGUF dynamic AOTI repeat outputs without promotion.

Unchanged packages and kernels, independent snapshots and explicit labels.
No tolerance is relaxed. This diagnostic measures no performance ranking.
"""
import argparse,datetime,fcntl,hashlib,importlib.util,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(16*1024*1024):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['et-source','gguf','precision-control','source-report','output','lock']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--deadline-utc',required=True);p.add_argument('--owned-worker',action='store_true')
    a=p.parse_args();cutoff=datetime.datetime.fromisoformat(a.deadline_utc.replace('Z','+00:00'))
    if not a.owned_worker:
        assert not a.output.exists();a.output.mkdir(parents=True)
        (a.output/'lock-owner.json').write_text(json.dumps(dict(pid=os.getpid()))+'\n')
        (a.output/'progress.json').write_text(json.dumps(dict(status='waiting-GPU-lock',controller_pid=os.getpid()))+'\n')
        with a.lock.open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            remaining=(cutoff-datetime.datetime.now(datetime.timezone.utc)).total_seconds();assert remaining>0
            with (a.output/'worker.log').open('x') as log:
                child=subprocess.Popen([sys.executable,'-B',str(Path(__file__).resolve()),*sys.argv[1:],'--owned-worker'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                try:assert child.wait(timeout=min(900,remaining))==0
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise
            end=time.monotonic()+60
            while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():
                assert time.monotonic()<end;time.sleep(1)
        return
    assert json.loads((a.output/'lock-owner.json').read_text())['pid']==os.getppid()
    os.environ.update(CUDA_VISIBLE_DEVICES='0',CUDA_HOME='/usr/local/cuda',TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',TORCH_CUDA_ARCH_LIST='12.0',OMP_NUM_THREADS='2',TORCHINDUCTOR_COMPILE_THREADS='1',MAX_JOBS='1',TRITON_CACHE_DIR=str(a.output/'triton-cache'))
    sys.path.insert(0,str(a.et_source.parent))
    import torch,triton
    from gguf import GGUFReader,GGMLQuantizationType
    from executorch.examples.models.gemma4_31b.batching_gguf_loader import decode_q4_blocks
    from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor
    from executorch.extension.llm.export.gguf import ExportableGGUFTensor
    from executorch.examples.models.gemma4_31b import cuda_packers
    from executorch.backends.cuda.triton.kernels import int4_quantized_gemm as native
    from aoti_single_threaded import SingleThreadedPackage
    from candidates_w4a16_v9.validate import precision_oracle
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
    assert torch.cuda.get_device_capability()==(12,0)
    assert digest(Path(native.__file__))=='fb72d31e6d3f91e076fe9721744eacb7c379ffe0612c57e3c48ae4a25c20027c'
    assert digest(a.precision_control)=='10affaf3cd36f17a68f8336e764e24b16e54dd509d89577c9009445b2f41cf9b'
    assert digest(Path(cuda_packers.__file__))=='c1243e267d373490cf325d14d9431a80d163c313b56d6f217c83cd7f1a74f15e'
    spec=importlib.util.spec_from_file_location('unchanged_precision_reference',a.precision_control);corrected=importlib.util.module_from_spec(spec);spec.loader.exec_module(corrected)
    assert digest(a.gguf)=='7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a'
    original=json.loads(a.source_report.read_text());case=original['AOTI_cases'][-1]
    assert case['role']=='Gemma-real-ffn_down' and case['bucket']==8 and case['config_id']=='v9_m8_down_native_sp4_r32'
    assert not case['tail_numerics'] and (case['n'],case['k'])==(3840,15360)
    report=dict(source_report_sha256=digest(a.source_report),GGUF_sha256=digest(a.gguf),original_failure_preserved=True,
                source_commit=json.loads((Path(__file__).parent/'source-commit.json').read_text()),
                numeric_threshold_unchanged=.005,no_performance_claim=True,no_candidate_promotion=True,
                original_config=case,controls=[],packages={})
    def save(status,**fields):
        (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        t=a.output/'progress.pending.json';t.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),**fields),indent=2)+'\n');t.replace(a.output/'progress.json')
    def stats(y,ref):
        d=y.float()-ref.float()
        return dict(finite=bool(torch.isfinite(y).all()),relative_rms=float(d.square().mean().sqrt()/ref.float().square().mean().sqrt().clamp_min(1e-8)),max_abs=float(d.abs().max()))
    def bounded():assert datetime.datetime.now(datetime.timezone.utc)<cutoff
    try:
        with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):
            t=next(t for t in GGUFReader(str(a.gguf)).tensors if t.name=='blk.0.ffn_down.weight')
            n,k=tuple(reversed(t.shape.tolist()));assert (n,k)==(3840,15360)
            raw=torch.frombuffer(memoryview(t.data),dtype=torch.uint8).clone()
            if t.tensor_type==GGMLQuantizationType.Q4_K:
                portable=ExportableGGUFTensor.from_raw(raw.reshape(n,-1),'q4_k')
                module=torch.nn.Linear(k,n,bias=False,device='meta')
                cuda_packers.pack_linear_for_cuda(module,{'weight':portable});obj=module.weight
            else:
                assert t.tensor_type in [GGMLQuantizationType.Q4_0,GGMLQuantizationType.Q4_1]
                typ='q4_0' if t.tensor_type==GGMLQuantizationType.Q4_0 else 'q4_1'
                portable=decode_q4_blocks(raw,typ,(n,k))
                obj=CudaCoalescedInt4Tensor.from_exportable_int4_tensor(portable)
            report['source_tensor']=dict(name=t.name,quantization=t.tensor_type.name,n=n,k=k,
                raw_weight_sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
                packer_sha256=digest(Path(cuda_packers.__file__)),original_benchmark_packing_branch_reproduced=True)
            w=tuple(getattr(obj,key).cuda() for key in ['qdata','scale','scale_step','zero_point','zero_point_step'])
            x=torch.randn((8,15360),dtype=torch.bfloat16,device='cuda',generator=torch.Generator(device='cuda').manual_seed(22000+8+3840))
            report['packed_weight_sha256']=[hashlib.sha256(v.cpu().view(torch.uint8).numpy().tobytes()).hexdigest() for v in w]
            models={}
            for label,proof in case['packages'].items():
                package=Path(proof['path']);assert digest(package)==proof['sha256'] and package.stat().st_size==proof['bytes']
                models[label]=SingleThreadedPackage(package,a.output/'package-loads'/label)
                report['packages'][label]=dict(sha256=proof['sha256'],extracted_members=models[label].members)
            cfg=case['reference_metadata']['config'];ncfg=triton.Config({k:v for k,v in cfg.items() if k not in ['num_warps','num_stages']},num_warps=cfg['num_warps'],num_stages=cfg['num_stages'])
            for m in [5,7,8]:
                xx=x[:m].contiguous();oracle=precision_oracle(xx,w)
                for label in ['reference','candidate','eager-native']:
                    bounded();save('running-repeat-diagnostic',m=m,label=label)
                    def fn():
                        if label=='eager-native':return corrected._launch(8,xx,*w,32,split_k=case['reference_metadata']['split_k'],config=ncfg)
                        return models[label](xx,*w)[0]
                    snapshots=[];pointers=[];checks=[]
                    for repeat in range(20):
                        y=fn();torch.cuda.synchronize();pointers.append(y.data_ptr());snapshots.append(y.clone())
                        checks.append(stats(snapshots[-1],oracle));torch.cuda.synchronize()
                    differences=[dict(repeat=i,**stats(y,snapshots[0]),different_elements=int((y!=snapshots[0]).sum())) for i,y in enumerate(snapshots)]
                    untouched=[fn() for _ in range(4)];torch.cuda.synchronize()
                    gate=[]
                    for y in snapshots:
                        try:torch.testing.assert_close(y.float(),oracle.float(),rtol=.05,atol=4.0);gate.append(True)
                        except AssertionError:gate.append(False)
                    record=dict(m=m,label=label,independent_snapshots=20,all_exact_equal=all(torch.equal(y,snapshots[0]) for y in snapshots[1:]),
                                numeric_checks=checks,pairwise_differences=differences,original_style_exact_equal=all(torch.equal(untouched[1],y) for y in untouched[2:]),
                                raw_output_pointers=pointers,elementwise_gate_all_pass=all(gate),RMS_gate_all_pass=all(v['finite'] and v['relative_rms']<.005 for v in checks))
                    report['controls'].append(record);save('repeat-diagnostic-recorded',m=m,label=label)
                    del snapshots,untouched,oracle
                    oracle=precision_oracle(xx,w)
            for model in models.values():model.close()
            save('completed',diagnostic_only=True)
    except Exception as e:
        (a.output/'exception-traceback.txt').write_text(traceback.format_exc());save('failed',detail=str(e)[-2000:]);raise

if __name__=='__main__':main()
