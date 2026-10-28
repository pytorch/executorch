"""Measure executor routing, staging and sampled-token waits without extra fences.

This prepared-token mini diagnostic preserves the model, cache and routing.
Host launch time and the subsequent GPU wait are reported separately; neither
is a standalone measurement of GPU model or sampler kernel time.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

PROFILER = r'''
struct ExecutorPhases {
  std::map<std::string,std::pair<uint64_t,uint64_t>> rows;
  ~ExecutorPhases(){
    nlohmann::json result=nlohmann::json::object();
    for(const auto& row:rows)result[row.first]={{"calls",row.second.first},{"nanoseconds",row.second.second}};
    std::fprintf(stderr,"SILICONBENCH_EXECUTOR_PHASES %s\n",result.dump().c_str());
  }
};
thread_local ExecutorPhases executor_phases;
struct ExecutorPhaseScope {
  std::string name;std::chrono::steady_clock::time_point begin;bool active=true;
  explicit ExecutorPhaseScope(std::string value):name(std::move(value)),begin(std::chrono::steady_clock::now()){}
  void stop(){if(!active)return;active=false;auto& row=executor_phases.rows[name];++row.first;
    row.second+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();}
  ~ExecutorPhaseScope(){stop();}
};
'''


def main(args):
    job, base, previous = args.job, args.base, args.routing_job
    assert job.is_dir() and not (job / 'executor-mini').exists()
    rc = 1
    try:
        assert json.loads((previous / 'exit-receipt.json').read_text())['returncode'] == 0
        sdk = base / 'MG-mini-required-side-binding-native-release-20261027-v2/native/executorch'
        original = sdk / 'backends/cuda/batching/cuda_executor.cpp'
        source = original.read_text()
        source = source.replace('#include <algorithm>', '#include <algorithm>\n#include <chrono>\n#include <map>\n#include <cstdio>\n#include <nlohmann/json.hpp>')
        marker = 'using ::executorch::extension::make_tensor_ptr;'
        assert source.count(marker) == 1
        source = source.replace(marker, PROFILER + '\n' + marker)
        marker = '  Result<llm_batching::util::PackedStep> step = sessions_.pack(batch);'
        assert source.count(marker) == 1
        source = source.replace(marker, '  Result<llm_batching::util::PackedStep> step = [&]{ExecutorPhaseScope phase("session_pack");return sessions_.pack(batch);}();')
        marker = '  for (const auto& slice : plan_slices(total, methods_)) {'
        assert source.count(marker) == 1
        source = source.replace(marker, '  auto planned_slices = [&]{ExecutorPhaseScope phase("plan_slices");return plan_slices(total, methods_);}();\n  for (const auto& slice : planned_slices) {')
        start = source.index('bool CudaExecutor::execute_raw(')
        end = source.index('std::optional<std::vector<Token>> CudaExecutor::argmax_rows', start)
        raw = source[start:end]
        marker = '    if (!ctl_->declare_step(std::vector<std::int32_t>('
        assert raw.count(marker) == 1
        raw = raw.replace(marker, '    ExecutorPhaseScope declaration_phase("declare_step");\n' + marker)
        marker = '    std::vector<std::int64_t> token_values('
        raw = raw.replace(marker, '    declaration_phase.stop();\n    ExecutorPhaseScope input_phase("input_staging");\n' + marker)
        marker = '    auto result = module_->execute(method.name, {tokens, positions, selector});'
        assert raw.count(marker) == 1
        raw = raw.replace(marker, '    input_phase.stop();\n    auto result = [&]{ExecutorPhaseScope phase("forward_launch:"+method.name);return module_->execute(method.name, {tokens, positions, selector});}();')
        marker = '    if (!consume({off, n, logits, taps ? &*taps : nullptr, indices})) {'
        assert raw.count(marker) == 1
        raw = raw.replace(marker, '    ExecutorPhaseScope consume_phase("output_consume_inclusive");\n' + marker)
        source = source[:start] + raw + source[end:]
        marker = '  const auto result = run_sampler();'
        assert source.count(marker) == 1
        source = source.replace(marker, '  const auto result = [&]{ExecutorPhaseScope phase("sampler_launch");return run_sampler();}();')
        start = source.index('std::optional<std::vector<Token>> CudaExecutor::run_device_sampler(')
        end = source.index('OffGraphKVMetrics CudaExecutor::kv_metrics()', start)
        sampler = source[start:end]
        marker = '  if (cudaMemcpyAsync('
        assert sampler.count(marker) == 1
        sampler = sampler.replace(marker, '  ExecutorPhaseScope wait_phase("sampled_tokens_copy_and_pending_GPU_wait");\n' + marker)
        source = source[:start] + sampler + source[end:]
        (job / 'profiled_cuda_executor.cpp').write_text(source)
        commands = json.loads((previous / 'build-commands.json').read_text())
        compile_cmd = commands[1][:]
        compile_cmd[compile_cmd.index('-o') + 1] = str(job / 'profiled_executor.o')
        compile_cmd[compile_cmd.index('-c') + 1] = str(job / 'profiled_cuda_executor.cpp')
        link = commands[-1][:]
        link[link.index('-o') + 1] = str(job / 'executor-mini')
        index = next(i for i, item in enumerate(link) if item.endswith('/libcuda_batching.a'))
        link.insert(index, str(job / 'profiled_executor.o'))
        (job / 'build-commands.json').write_text(json.dumps([compile_cmd, link], indent=2))
        (job / 'bindings.json').write_text(json.dumps(dict(
            original_executor=dict(path=str(original), SHA256=hashlib.sha256(original.read_bytes()).hexdigest()),
            profiled_executor_SHA256=hashlib.sha256(source.encode()).hexdigest(),
            same_routing_job=str(previous), artifact_bindings=json.loads((previous/'bindings.json').read_text()),
            no_extra_GPU_synchronization=True, mini_only=True), indent=2))
        with (job / 'build.log').open('x') as log:
            for cmd in [compile_cmd, link]:
                subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        artifact = base / 'four-opt-MG-mini81-stable-layout128-qualification-20261028-v1/same-PTE'
        env = dict(os.environ, LD_LIBRARY_PATH=str(sdk / 'cmake-out/lib'),
            ET_CUDA_RING_JOURNAL_BUFFERS='0', ET_CUDA_RING_ARCHIVE_BATCH_ROWS='64', ET_CUDA_SHARED_IDLE_REFRESH='1')
        rows = []
        with (base / 'gpu-benchmark.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'], text=True).strip()
            for round_id in range(1,5):
                for c in [1,2,4,8,16,32,64]:
                    folder = job / 'cases' / str(round_id) / str(c)
                    folder.mkdir(parents=True)
                    command = [str(job/'executor-mini'), '--pte='+str(artifact/'model.pte'),
                        '--data_path='+str(artifact/'aoti_cuda_blob.ptd'), '--concurrency='+str(c),
                        '--budget=4096','--prompt_tokens=1024','--prefix_entries=0','--cold_sessions=true',
                        '--admission_quiet_us=300','--admission_maximum_us=2000',
                        '--diagnostic_admission_barrier=false','--shared_idle_refresh=false']
                    (job/'progress.json').write_text(json.dumps(dict(round=round_id,c=c,complete=len(rows),prefix_entries=0,mini_only=True)))
                    with (folder/'stdout.json').open('x') as out, (folder/'stderr.log').open('x') as err:
                        subprocess.run(command, stdout=out, stderr=err, env=env, timeout=300, check=True)
                    v = json.loads((folder/'stdout.json').read_text())
                    assert all(group['actual_output_tokens_each']==64 for group in v['groups'])
                    rows.append(dict(round=round_id,c=c,directory=str(folder)))
                    (job/'records.json').write_text(json.dumps(rows,indent=2))
        (job/'result.json').write_text(json.dumps(dict(records=rows,mini_only=True,qualification=False,
            CPU_timings_not_GPU_kernel_times=True,full_main_unchanged=True),indent=2))
        rc = 0
    except BaseException:
        (job/'failure.txt').write_text(traceback.format_exc())
    finally:
        (job/'exit-receipt.json').write_text(json.dumps(dict(returncode=rc,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['base','job','routing-job']:
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
