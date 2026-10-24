"""Diagnose preserved native AOTI producer/reducer binding on its actual geometry.

Compare the original C API package, direct unchanged cubins, and eager native
launches. ELF requirements are 128 producer / 64 reducer threads; requested Triton
warp metadata alone is not used as executable geometry. No kernel or runtime
is patched. Record output and FP32 partial-buffer validity without promotion.
"""
import argparse,ctypes,datetime,importlib.util,fcntl,hashlib,json,os,signal,subprocess,sys,time,traceback,zipfile
from pathlib import Path

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

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
    sys.path.insert(0,str(a.et_source.parent))
    import torch
    from gguf import GGUFReader,GGMLQuantizationType
    from executorch.examples.models.gemma4_31b.batching_gguf_loader import decode_q4_blocks
    from executorch.backends.cuda.coalesced_int4_tensor import CudaCoalescedInt4Tensor
    from executorch.examples.models.gemma4_31b import cuda_packers
    from executorch.backends.cuda.triton.kernels import int4_quantized_gemm as native
    from candidates_w4a16_v9.validate import precision_oracle
    from aoti_single_threaded import SingleThreadedPackage
    assert torch.cuda.get_device_capability()==(12,0)
    assert digest(native.__file__)=='fb72d31e6d3f91e076fe9721744eacb7c379ffe0612c57e3c48ae4a25c20027c'
    assert digest(cuda_packers.__file__)=='c1243e267d373490cf325d14d9431a80d163c313b56d6f217c83cd7f1a74f15e'
    assert digest(a.gguf)=='7cef807d864aed1cbb9b93623df3724bb5d19a943fca01c56146dda78004233a'
    assert digest(a.source_report)=='ce1c8fc3cbbdcba7592e323e4bbd77b17719e69d8e608c1a90b778a3e66e381a'
    assert digest(a.precision_control)=='10affaf3cd36f17a68f8336e764e24b16e54dd509d89577c9009445b2f41cf9b'
    spec=importlib.util.spec_from_file_location('unchanged_eager_precision_native',a.precision_control)
    corrected=importlib.util.module_from_spec(spec);spec.loader.exec_module(corrected)
    case=json.loads(a.source_report.read_text())['AOTI_cases'][-1]
    assert case['config_id']=='v9_m8_down_native_sp4_r32' and case['reference_metadata']['split_k']==8
    proof=case['packages']['reference'];package=Path(proof['path']);assert digest(package)==proof['sha256']
    extracted=a.output/'unchanged-reference-cubins';extracted.mkdir()
    members={}
    with zipfile.ZipFile(package) as archive:
        wrapper=archive.read(next(x for x in archive.namelist() if x.endswith('.wrapper.cpp'))).decode()
        for m in archive.infolist():
            if not m.filename.endswith('.cubin'):continue
            name=Path(m.filename).name;raw=archive.read(m);(extracted/name).write_bytes(raw)
            members[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    report=dict(source_commit=json.loads((Path(__file__).parent/'source-commit.json').read_text()),
        original_failure_report_sha256=digest(a.source_report),original_reference_package_sha256=digest(package),
        extracted_cubins=members,native_kernel_and_weights_unchanged=True,numeric_threshold_unchanged=.005,
        no_performance_claim=True,no_candidate_promotion=True,direct_buffers_initialized_to_NaN=True,
        compiled_launch_geometry=dict(producer_threads=128,reducer_threads=64,producer_grid=[120,8,1],reducer_grid=[60,1,1]),controls=[])
    def save(status,**fields):
        (a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        q=a.output/'progress.pending.json';q.write_text(json.dumps(dict(status=status,controller_pid=os.getpid(),**fields),indent=2)+'\n');q.replace(a.output/'progress.json')
    driver=ctypes.CDLL('libcuda.so.1');ptr=ctypes.c_void_p
    def check(error,action):
        if error:raise RuntimeError('CUDA driver '+action+' failed: '+str(error))
    driver.cuModuleLoad.argtypes=[ctypes.POINTER(ptr),ctypes.c_char_p];driver.cuModuleLoad.restype=ctypes.c_int
    driver.cuModuleGetFunction.argtypes=[ctypes.POINTER(ptr),ptr,ctypes.c_char_p];driver.cuModuleGetFunction.restype=ctypes.c_int
    driver.cuModuleUnload.argtypes=[ptr];driver.cuModuleUnload.restype=ctypes.c_int
    driver.cuLaunchKernel.argtypes=[ptr,*([ctypes.c_uint]*7),ptr,ctypes.POINTER(ptr),ctypes.POINTER(ptr)];driver.cuLaunchKernel.restype=ctypes.c_int
    modules=[];functions={};model=None
    try:
        torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest')
        with torch.inference_mode(),torch.cuda.stream(torch.cuda.Stream()):
            for name in members:
                module=ptr();check(driver.cuModuleLoad(ctypes.byref(module),str(extracted/name).encode()),'module load');modules.append(module)
                for symbol in ['_int4_w4a16_splitk_kernel','_splitk_reduce_kernel']:
                    function=ptr()
                    if driver.cuModuleGetFunction(ctypes.byref(function),module,symbol.encode())==0:
                        assert symbol not in functions;functions[symbol]=function
            assert set(functions)=={'_int4_w4a16_splitk_kernel','_splitk_reduce_kernel'}
            model=SingleThreadedPackage(package,a.output/'original-CAPI-reload')
            cfg=case['reference_metadata']['config']
            import triton
            ncfg=triton.Config({k:v for k,v in cfg.items() if k not in ['num_warps','num_stages']},num_warps=cfg['num_warps'],num_stages=cfg['num_stages'])
            tensor=next(t for t in GGUFReader(str(a.gguf)).tensors if t.name=='blk.0.ffn_down.weight')
            assert tensor.tensor_type==GGMLQuantizationType.Q4_1
            n,k=tuple(reversed(tensor.shape.tolist()));assert (n,k)==(3840,15360)
            raw=torch.frombuffer(memoryview(tensor.data),dtype=torch.uint8).clone()
            obj=CudaCoalescedInt4Tensor.from_exportable_int4_tensor(decode_q4_blocks(raw,'q4_1',(n,k)))
            w=tuple(getattr(obj,key).cuda() for key in ['qdata','scale','scale_step','zero_point','zero_point_step'])
            x=torch.randn((8,k),dtype=torch.bfloat16,device='cuda',generator=torch.Generator(device='cuda').manual_seed(25848))
            report.update(source_tensor=dict(name=tensor.name,quantization='Q4_1',raw_sha256=hashlib.sha256(raw.numpy().tobytes()).hexdigest()),
                packed_weight_sha256=[hashlib.sha256(v.cpu().view(torch.uint8).numpy().tobytes()).hexdigest() for v in w])
            def launch(symbol,grid,warps,shared,args):
                addresses=(ptr*len(args))(*(ctypes.cast(ctypes.byref(x),ptr) for x in args))
                check(driver.cuLaunchKernel(functions[symbol],*grid,warps*32,1,1,shared,
                    ptr(torch.cuda.current_stream().cuda_stream),addresses,None),symbol)
            def stats(y,ref):
                d=y.float()-ref.float();finite=bool(torch.isfinite(y).all())
                return dict(finite=finite,unwritten_or_nonfinite_elements=int((~torch.isfinite(y)).sum()),
                    relative_rms=float(d.square().mean().sqrt()/ref.float().square().mean().sqrt().clamp_min(1e-8)) if finite else None,
                    max_abs=float(d.abs().max()) if finite else None)
            for m in [5,7,8]:
                xx=x[:m].contiguous();oracle=precision_oracle(xx,w)
                for label in ['original-CAPI','direct-unchanged-cubins','eager-native']:
                    assert datetime.datetime.now(datetime.timezone.utc)<cutoff
                    save('running-native-binding-control',m=m,label=label)
                    checks=[]
                    for repeat in range(20):
                        partial=None
                        if label=='original-CAPI':out=model(xx,*w)[0]
                        elif label=='eager-native':out=corrected._launch(8,xx,*w,32,split_k=8,config=ncfg)
                        else:
                            partial=torch.full((8,8,n),float('nan'),dtype=torch.float32,device='cuda')
                            out=torch.full((m,n),float('nan'),dtype=torch.bfloat16,device='cuda')
                            inputs=[ptr(t.data_ptr()) for t in (xx,*w,partial)]+[ctypes.c_int32(m),ptr(),ptr()]
                            launch('_int4_w4a16_splitk_kernel',(120,8,1),4,4096,inputs)
                            launch('_splitk_reduce_kernel',(60,1,1),2,0,[ptr(partial.data_ptr()),ptr(out.data_ptr()),ctypes.c_int32(m),ptr(),ptr()])
                        torch.cuda.synchronize();result=stats(out,oracle)
                        passed=True
                        try:torch.testing.assert_close(out.float(),oracle.float(),rtol=.05,atol=4.0)
                        except AssertionError:passed=False
                        result.update(repeat=repeat,elementwise_gate_passed=passed)
                        if partial is not None:
                            active=partial[:,:m]
                            sum_native_partials=active.sum(0).bfloat16()
                            result.update(producer_active_partial_elements=8*m*n,
                                producer_unwritten_active_partial_elements=int((~torch.isfinite(active)).sum()),
                                direct_output_vs_FP32_partial_sum=stats(out,sum_native_partials),
                                FP32_partial_sum_vs_oracle=stats(sum_native_partials,oracle))
                        checks.append(result)
                    report['controls'].append(dict(m=m,label=label,producer_threads=128,reducer_threads=64,
                        expected_elements=m*n,checks=checks,RMS_gate_all_pass=all(v['finite'] and v['relative_rms']<.005 for v in checks),
                        elementwise_gate_all_pass=all(v['elementwise_gate_passed'] for v in checks)))
                    save('native-binding-control-recorded',m=m,label=label)
            model.close();model=None
            save('completed',diagnostic_only=True,no_numerical_acceptance_claim=True)
    except Exception as e:
        (a.output/'exception-traceback.txt').write_text(traceback.format_exc());save('failed',detail=str(e)[-1600:]);raise
    finally:
        if model is not None:model.close()
        for module in modules:check(driver.cuModuleUnload(module),'module unload')

if __name__=='__main__':main()
