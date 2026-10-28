"""Profile native mini routing with fixed 64-token outputs and prefix OFF."""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback


PHASE_PROFILER = r'''
struct RoutingPhases {
  std::map<std::string,std::pair<uint64_t,uint64_t>> rows;
  ~RoutingPhases(){
    nlohmann::json result=nlohmann::json::object();
    for(const auto& row:rows)result[row.first]={{"calls",row.second.first},{"nanoseconds",row.second.second}};
    std::fprintf(stderr,"SILICONBENCH_ROUTING_PHASES %s\n",result.dump().c_str());
  }
};
thread_local RoutingPhases routing_phases;
struct RoutingPhaseScope {
  const char* name;std::chrono::steady_clock::time_point begin;
  explicit RoutingPhaseScope(const char* value):name(value),begin(std::chrono::steady_clock::now()){}
  ~RoutingPhaseScope(){auto& row=routing_phases.rows[name];++row.first;
    row.second+=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-begin).count();}
};
'''


def main(args):
    job, base = args.job, args.base
    assert job.is_dir() and not (job / "routing-mini").exists()
    rc = 1

    def state(status, **extra):
        (job / "progress.json").write_text(json.dumps(dict(status=status,
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            prefix_entries=0, mini_only=True, **extra)))

    try:
        mini = base / "four-opt-relocatable-mini64-default-chunk-build-20261028-v2"
        observed = base / "four-opt-MG-observable-full-native-worker-20261028-v5"
        stable = base / "four-opt-MG-mini81-stable-layout128-qualification-20261028-v1"
        source = (mini / "admission64.cpp").read_text()
        source = source.replace("FLAGS_concurrency==1||FLAGS_concurrency==8", "FLAGS_concurrency==1||FLAGS_concurrency==2||FLAGS_concurrency==4||FLAGS_concurrency==8")
        source = source.replace("for(int n=0;n<FLAGS_prompt_tokens;++n)histories[owner].push_back", "histories[owner].clear();for(int n=0;n<FLAGS_prompt_tokens;++n)histories[owner].push_back")
        source = source.replace('histories[owner].insert(histories[owner].end(),ids.begin(),ids.end());outputs.push_back(ids);', 'outputs.push_back(ids);')
        marker = '    }\n    runtime.shutdown();'
        assert source.count(marker) == 1
        source = source.replace(marker, '      for(int owner=0;owner<FLAGS_concurrency;++owner)require(!runtime.close_session_async("owner"+std::to_string(owner)).get(),"close fixed-input group");\n' + marker)
        trace_start = source.index('      Json batch=Json::array();')
        trace_end = source.index('      trace_.push_back(std::move(batch));', trace_start) + len('      trace_.push_back(std::move(batch));')
        source = source[:trace_start] + '      // Composition is measured in Runner; avoid per-task JSON allocation here.' + source[trace_end:]
        (job / "routing_mini.cpp").write_text(source)

        runner = (observed / "observed_runner.cpp").read_text()
        runner = runner.replace("void RunnerImpl::run_() {", PHASE_PROFILER + "\nvoid RunnerImpl::run_() {")
        functions = ["process_pending_commands_", "coalesce_admission_", "reap_cancelled_"]
        for name in functions:
            marker = "void RunnerImpl::" + name + "() {"
            assert runner.count(marker) == 1
            runner = runner.replace(marker, marker + '\n  RoutingPhaseScope phase("' + name + '");')
        marker = "bool RunnerImpl::execute_one_batch_() {"
        runner = runner.replace(marker, marker + '\n  RoutingPhaseScope phase("batch_total_inclusive");')
        marker = "    tasks = scheduler_->get_work();"
        runner = runner.replace(marker, '    {RoutingPhaseScope phase("scheduler_get_work");' + marker + "}")
        marker = "  const bool ok = executor_.execute(batch, out);"
        assert runner.count(marker) == 1
        runner = runner.replace(marker, '  bool ok;{RoutingPhaseScope phase("executor_execute");ok=executor_.execute(batch,out);}')
        marker = "      handle_output_(input.sid, std::move(out.outputs[i]));"
        runner = runner.replace(marker, '{RoutingPhaseScope phase("output_and_decode_resubmit");' + marker + "}")
        runner = runner.replace('#include <algorithm>', '#include <algorithm>\n#include <map>')
        (job / "routing_runner.cpp").write_text(runner)
        commands = json.loads((mini / "build-commands.json").read_text())
        compile_mini = [item.replace(str(mini / "admission64.cpp.o"), str(job / "routing-mini.o")).replace(str(mini / "admission64.cpp"), str(job / "routing_mini.cpp")) for item in commands[0]]
        compile_runner = json.loads((observed / "build-commands.json").read_text())[0]
        compile_runner = [item.replace(str(observed / "0-runner.o"), str(job / "routing-runner.o")).replace(str(observed / "observed_runner.cpp"), str(job / "routing_runner.cpp")) for item in compile_runner]
        link = commands[1]
        link = [str(job / "routing-runner.o") if item.endswith('/libextension_llm_runner.a') else item for item in link]
        link = [str(stable / "libaoti_cuda_backend.a") if item.endswith('/libaoti_cuda_backend.a') else item for item in link]
        link = [item.replace(str(mini / "admission64.cpp.o"), str(job / "routing-mini.o")).replace(str(mini / "admission64-mini"), str(job / "routing-mini")) for item in link]
        (job / "build-commands.json").write_text(json.dumps([compile_mini, compile_runner, link], indent=2))
        state("building-isolated-native-routing-profiler")
        with (job / "build.log").open("x") as log:
            for command in [compile_mini, compile_runner, link]:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        artifact = stable / "same-PTE"
        bindings = {name:dict(path=str(artifact / name), SHA256=hashlib.sha256((artifact / name).read_bytes()).hexdigest())
                    for name in ['model.pte','aoti_cuda_blob.ptd']}
        bindings['native_archive'] = dict(path=str(stable / "libaoti_cuda_backend.a"), SHA256=hashlib.sha256((stable / "libaoti_cuda_backend.a").read_bytes()).hexdigest())
        (job / "bindings.json").write_text(json.dumps(bindings, indent=2))
        env = dict(os.environ, LD_LIBRARY_PATH=str(base / "MG-mini-required-side-binding-native-release-20261027-v2/native/executorch/cmake-out/lib"),
            ET_CUDA_RING_JOURNAL_BUFFERS="0", ET_CUDA_RING_ARCHIVE_BATCH_ROWS="64", ET_CUDA_SHARED_IDLE_REFRESH="1")
        records = []
        modes = [('current',300,2000,False), ('no-coalescing',0,0,False), ('admission-barrier-reference',0,0,True)]
        with (base / "gpu-benchmark.lock").open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
            for round_id in range(1,5):
                for concurrency in [1,2,4,8,16,32,64]:
                    for name, quiet, maximum, barrier in (modes if round_id % 2 else list(reversed(modes))):
                        state("profiling-prefix-OFF-routing", round=round_id, c=concurrency, mode=name, complete=len(records))
                        folder = job / "cases" / str(round_id) / str(concurrency) / name
                        folder.mkdir(parents=True)
                        command = [str(job / "routing-mini"), '--pte='+str(artifact/'model.pte'), '--data_path='+str(artifact/'aoti_cuda_blob.ptd'),
                            '--concurrency='+str(concurrency), '--budget=4096','--prompt_tokens=1024','--prefix_entries=0','--cold_sessions=true',
                            '--admission_quiet_us='+str(quiet),'--admission_maximum_us='+str(maximum),
                            '--diagnostic_admission_barrier='+str(barrier).lower(),'--shared_idle_refresh=false']
                        with (folder/'stdout.json').open('x') as out, (folder/'stderr.log').open('x') as err:
                            subprocess.run(command, stdout=out, stderr=err, env=env, check=True, timeout=300)
                        result=json.loads((folder/'stdout.json').read_text())
                        assert result['prefix_entries']==0 and result['configured_active_sessions']==64
                        assert all(group['actual_output_tokens_each']==64 for group in result['groups'])
                        records.append(dict(round=round_id,c=concurrency,mode=name,directory=str(folder)))
                        (job/'records.json').write_text(json.dumps(records,indent=2))
        (job/'result.json').write_text(json.dumps(dict(records=records,mini_only=True,independent_review_required=True,
            current_main_unchanged=True,admission_barrier_is_reference_only=True),indent=2))
        state("routing-profile-complete-needs-independent-review", complete=len(records))
        rc=0
    except BaseException:
        (job/'failure.txt').write_text(traceback.format_exc())
        state("routing-profile-failed-preserved")
    finally:
        (job/'exit-receipt.json').write_text(json.dumps(dict(returncode=rc,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--base',type=Path,required=True)
    main(parser.parse_args())
