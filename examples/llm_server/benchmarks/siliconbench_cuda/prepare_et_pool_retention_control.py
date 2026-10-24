"""Build an isolated ET pool-retention diagnostic with the existing model ABI.

Uses the existing allocator API and audited native libraries. It changes no
weight, export, kernel, static method, or installed server. Performance and
correctness acceptance are separate from preparation.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


DIAGNOSTIC = r'''
namespace {
void check_cuda(cudaError_t result) {
  if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

void memory_snapshot(const char* stage) {
  check_cuda(cudaDeviceSynchronize());
  size_t free_bytes = 0, total_bytes = 0;
  check_cuda(cudaMemGetInfo(&free_bytes, &total_bytes));
  auto pool = executorch::extension::cuda::CudaAllocator::pool_for_device(0);
  uint64_t reserved = 0, used = 0, threshold = 0;
  if (pool != nullptr) {
    check_cuda(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved));
    check_cuda(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used));
    check_cuda(cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
  }
  std::cerr << "ET_POOL_MEMORY " << nlohmann::json({
    {"stage", stage}, {"free_bytes", free_bytes}, {"total_bytes", total_bytes},
    {"private_pool_present", pool != nullptr}, {"private_pool_reserved_bytes", reserved},
    {"private_pool_used_bytes", used}, {"private_pool_release_threshold_bytes", threshold}
  }).dump() << '\n';
}
} // namespace
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['et-source', 'recorded-et-source', 'worker-source', 'build-inputs',
                 'base-launcher', 'base-wrapper', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    for name in ['worker-source-sha256', 'build-audit-sha256',
                 'base-launcher-sha256', 'base-wrapper-sha256']:
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--base-worker-reference', required=True)
    parser.add_argument('--python', required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Preserve earlier controls and results'
    args.output.mkdir(parents=True)

    def state(status, **fields):
        pending = args.output/'progress.pending.json'
        pending.write_text(json.dumps(dict(status=status, controller_pid=os.getpid(),
                                          model_acceptance_pending=True, **fields), indent=2)+'\n')
        pending.replace(args.output/'progress.json')

    try:
        inputs = args.build_inputs.resolve()
        audit_path = inputs/'build-audit.json'
        pinned = [(args.worker_source, args.worker_source_sha256),
                  (audit_path, args.build_audit_sha256),
                  (args.base_launcher, args.base_launcher_sha256),
                  (args.base_wrapper, args.base_wrapper_sha256)]
        for path, expected in pinned:
            assert digest(path) == expected, str(path)
        tracked = [path for path, _ in pinned] + sorted((inputs/'lib').glob('*'))
        before = {str(path): digest(path) for path in tracked if path.is_file()}
        allocator = inputs/'include/executorch/extension/cuda/cuda_allocator.h'
        assert 'pool_for_device' in allocator.read_text()
        assert 'release_cached_memory' in allocator.read_text()
        state('preparing-isolated-worker')
        text = args.worker_source.read_text()
        text = replace_once(text, '#include <gflags/gflags.h>',
                            '#include <gflags/gflags.h>\n'
                            '#include <executorch/extension/cuda/cuda_allocator.h>\n'
                            '#include <chrono>\n#include <thread>\n#include <cuda_runtime_api.h>')
        text = replace_once(text, 'int main(int argc, char** argv) {',
                            'DEFINE_int64(pool_retain_mib, -1, "Private pool retention MiB; -1 keeps original policy");\n'
                            'DEFINE_bool(trim_pool_at_eof, false, "Diagnostic trim after runtime shutdown, before executor destruction");\n'
                            + DIAGNOSTIC + '\nint main(int argc, char** argv) {')
        text = replace_once(text, 'gflags::ParseCommandLineFlags(&argc, &argv, true);',
                            'gflags::ParseCommandLineFlags(&argc, &argv, true);\n'
                            '  if (FLAGS_pool_retain_mib < -1 || FLAGS_pool_retain_mib > 1048576) return 1;')
        marker = '    executorch::examples::llm_server::MultiplexedWorkerConfig transport;'
        text = replace_once(text, marker, r'''
    const auto startup_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(120);
    while (!runtime.info().ready && std::chrono::steady_clock::now() < startup_deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if (!runtime.info().ready) throw std::runtime_error("Diagnostic runtime initialization failed");
    memory_snapshot("ready-before-policy");
    auto private_pool = executorch::extension::cuda::CudaAllocator::pool_for_device(0);
    if (FLAGS_pool_retain_mib >= 0) {
      if (private_pool == nullptr) throw std::runtime_error("No private CUDA pool to control");
      const uint64_t retain_bytes = static_cast<uint64_t>(FLAGS_pool_retain_mib) * 1024 * 1024;
      check_cuda(cudaMemPoolSetAttribute(private_pool, cudaMemPoolAttrReleaseThreshold, &retain_bytes));
    }
    memory_snapshot("ready-after-policy");
''' + marker)
        text = replace_once(text, 'runtime, STDIN_FILENO, output, std::move(transport));',
                            'runtime, STDIN_FILENO, output, std::move(transport));\n'
                            '    memory_snapshot("after-runtime-shutdown-executor-live");\n'
                            '    if (FLAGS_trim_pool_at_eof) {\n'
                            '      executorch::extension::cuda::CudaAllocator::release_cached_memory(0);\n'
                            '      memory_snapshot("after-idle-trim-executor-live");\n'
                            '    }')
        source = args.output/'et_pool_retention_worker.cpp'
        source.write_text(text)
        py = args.base_launcher.read_text()
        py = replace_once(py, 'import argparse', 'import argparse\nimport os')
        py = replace_once(py, 'f"--prefix_cache_entries={args.prefix_cache_entries}"]',
                          'f"--prefix_cache_entries={args.prefix_cache_entries}",\n'
                          '        f"--pool_retain_mib={args.pool_retain_mib}",\n'
                          '        f"--trim_pool_at_eof={str(args.trim_pool_at_eof).lower()}"]')
        py = replace_once(py, 'parser.add_argument("--prefix-cache-entries", type=int, default=16)',
                          'parser.add_argument("--prefix-cache-entries", type=int, default=16)\n'
                          '    parser.add_argument("--pool-retain-mib", type=int, default=int(os.environ.get("SB_ET_POOL_RETAIN_MIB", "-1")))\n'
                          '    parser.add_argument("--trim-pool-at-eof", action="store_true", default=os.environ.get("SB_ET_TRIM_POOL_AT_EOF", "0") == "1")')
        ast.parse(py)
        launcher = args.output/'serve_et_pool_retention.py'
        launcher.write_text(py)
        wrapper = args.base_wrapper.read_text()
        wrapper = replace_once(wrapper, str(args.base_launcher), str(launcher.resolve()))
        binary = args.output/'et_pool_retention_worker'
        wrapper = replace_once(wrapper, args.base_worker_reference, shlex.quote(str(binary.resolve())))
        generated_wrapper = args.output/'serve-pool-control.sh'
        generated_wrapper.write_text(wrapper)
        generated_wrapper.chmod(0o700)
        subprocess.run(['bash', '-n', str(generated_wrapper)], check=True, timeout=30)
        audit = json.loads(audit_path.read_text())
        recorded = str(args.recorded_et_source)
        et = str(args.et_source.resolve())
        compile_command = audit['compile_command'].copy()
        for i, arg in enumerate(compile_command):
            compile_command[i] = arg.replace(recorded+'/cmake-out/include', str(inputs/'include')).replace(
                recorded+'/cmake-out/third-party/gflags/include', str(inputs/'gflags-include')).replace(recorded, et)
        obj = args.output/'worker.o'
        compile_command[compile_command.index('-c')+1] = str(source.resolve())
        compile_command[compile_command.index('-o')+1] = str(obj.resolve())
        link_command = audit['link_command'].copy()
        assert sum(arg.endswith('.o') for arg in link_command) == 1
        for i, arg in enumerate(link_command):
            if arg.endswith('.o'):
                link_command[i] = str(obj.resolve())
            elif arg.startswith('lib') and arg.endswith('.a'):
                link_command[i] = str(inputs/'lib'/arg)
            elif recorded+'/cmake-out/lib/' in arg or recorded+'/cmake-out/third-party/gflags/libgflags_' in arg:
                start = arg.index(recorded)
                end = arg.find(',', start)
                if end < 0:
                    end = len(arg)
                link_command[i] = arg[:start]+str(inputs/'lib'/Path(arg[start:end]).name)+arg[end:]
            elif arg.startswith('-Wl,-rpath,'):
                link_command[i] = '-Wl,-rpath,'+str(inputs/'lib')+':/usr/local/cuda/lib64'
            elif arg.startswith('/usr/local/cuda-13.2/') and not Path(arg).exists():
                link_command[i] = arg.replace('/usr/local/cuda-13.2/', '/usr/local/cuda/')
        link_command[link_command.index('-o')+1] = str(binary.resolve())
        state('building-isolated-worker-no-GPU-work')
        with (args.output/'build.log').open('x') as log:
            subprocess.run(compile_command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
            subprocess.run(link_command, cwd=args.output, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
        result = subprocess.run([str(binary.resolve()), '--helpshort'], capture_output=True, text=True, timeout=30)
        assert 'pool_retain_mib' in result.stdout+result.stderr
        subprocess.run([args.python, '-B', str(launcher), '--help'], capture_output=True, text=True, check=True, timeout=60)
        assert before == {str(path):digest(path) for path in tracked if path.is_file()}
        provenance = dict(worker_binary=str(binary.resolve()), wrapper=str(generated_wrapper.resolve()),
                          binary_sha256=digest(binary), worker_source_sha256=digest(source),
                          launcher_sha256=digest(launcher), wrapper_sha256=digest(generated_wrapper),
                          original_file_hashes=before, compile_command=compile_command, link_command=link_command,
                          original_source_and_libraries_unchanged=True, no_model_loaded=True,
                          safe_policy_set_before_transport_admission=True,
                          optional_trim_only_after_runtime_shutdown_and_device_synchronize=True,
                          no_per_forward_synchronization_added=True,
                          CUDA_graphs_weights_KV_layout_context_and_scheduler_unchanged=True,
                          no_performance_or_correctness_acceptance=True)
        (args.output/'build-provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
        state('completed', isolated_worker_prepared=True)
    except Exception as exc:
        state('failed', error_type=type(exc).__name__, detail=str(exc)[-1200:])
        raise


if __name__ == '__main__':
    main()
