"""Add read-only logical KV dumps to a private copy of the actual native probe.

The original SDK, archive, PTE, PTD and binaries are preserved. The diagnostic
adds no virtual methods or fields. Its dump operation runs only between steps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        while b := f.read(16 << 20): h.update(b)
    return h.hexdigest()


METHOD = r'''
  bool diagnostic_dump_logical_kv(int32_t sid, const char* directory) {
    std::lock_guard<std::mutex> guard(mutex_);
    const auto found = seqs_.find(sid);
    if (found == seqs_.end() || inflight_ || !pool_.allocated()) return false;
    const auto device = CUDAGuard::create(pool_.device());
    if (!device.ok() || pool_.synchronize(true) != Error::Ok) return false;
    const int column = found->second.column;
    const int length = found->second.plan.length();
    // Deliberately restricted to the first 64 positions, before any wrap.
    if (length <= 0 || length > 64) return false;
    std::ofstream meta(std::string(directory) + "/metadata.json");
    if (!meta.good()) return false;
    meta << "{\"sid\":" << sid << ",\"length\":" << length
         << ",\"column\":" << column << ",\"columns\":" << columns_
         << ",\"split_ring\":" << (cfg_.split_ring_history ? "true" : "false")
         << ",\"layers\":[";
    for (size_t layer = 0; layer < geometry_.layers.size(); ++layer) {
      const auto& geometry = geometry_.layers[layer];
      const size_t row_bytes = geometry.n_kv_heads * geometry.head_dim * 2;
      const bool ring = is_ring(geometry);
      const int64_t physical_rows = ring ?
          std::min(layer_rows_[layer], pool_.layer_allocated_rows(layer) / columns_) :
          pool_.sequence_capacity(column);
      if (physical_rows < length) return false;
      if (layer) meta << ",";
      meta << "{\"layer\":" << layer << ",\"kv_heads\":" << geometry.n_kv_heads
           << ",\"head_dim\":" << geometry.head_dim << ",\"rows\":" << physical_rows
           << ",\"ring\":" << (ring ? "true" : "false") << "}";
      for (bool value : {false, true}) {
        std::vector<uint8_t> host(static_cast<size_t>(length) * row_bytes);
        auto* base = static_cast<uint8_t*>(pool_.layer_data(layer, value));
        for (int pos = 0; pos < length; ++pos) {
          const int64_t row = ring ? (pos % physical_rows) * columns_ + column :
              pool_.sequence_offset(column) + pos;
          if (cudaMemcpy(host.data() + pos * row_bytes, base + row * row_bytes,
                         row_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        }
        const std::string name = std::string(directory) + "/layer-" +
            std::to_string(layer) + (value ? "-V.bf16" : "-K.bf16");
        std::ofstream file(name, std::ios::binary);
        file.write(reinterpret_cast<const char*>(host.data()), host.size());
        if (!file.good()) return false;
      }
    }
    meta << "]}\n";
    return meta.good();
  }
'''

EXPORT = r'''
extern "C" bool siliconbench_diagnostic_dump_logical_kv(
    cache::BatchControl* control, int32_t sid, const char* directory) {
  auto* actual = dynamic_cast<CudaBatchedSequenceCache*>(control);
  return actual != nullptr && actual->diagnostic_dump_logical_kv(sid, directory);
}
'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native-parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    parent = args.native_parent
    sdk = parent / 'native/executorch'
    original_cpp = sdk / 'backends/cuda/runtime/cuda_kv_cache.cpp'
    original_probe = parent / 'mini_probe.cpp'
    archive = sdk / 'cmake-out/lib/libaoti_cuda_backend.a'
    controls = [original_cpp, original_probe, archive, parent / 'probe-build/rewind_kv_probe']
    before = {str(f): digest(f) for f in controls}
    cache = original_cpp.read_text()
    start = cache.index('class CudaBatchedSequenceCache final')
    anchor = cache.index('  // -- CacheControl / BatchControl.', start)
    cache = cache[:anchor] + METHOD + '\n' + cache[anchor:]
    cache = cache.replace('#include <cstring>', '#include <cstring>\n#include <fstream>')
    anchor = cache.rindex('} // namespace executorch::backends::cuda')
    cache = cache[:anchor] + EXPORT + '\n' + cache[anchor:]
    copy_cpp = args.output / 'cuda_kv_cache.cpp'
    copy_cpp.write_text(cache)
    probe = original_probe.read_text()
    anchor = 'namespace cb = executorch::backends::cuda::batching;'
    assert probe.count(anchor) == 1
    probe = probe.replace(anchor,
        'extern "C" bool siliconbench_diagnostic_dump_logical_kv(\n'
        '    executorch::extension::llm::cache::BatchControl*, int32_t, const char*);\n' + anchor)
    anchor = '      } else if (op == "step") {'
    assert probe.count(anchor) == 1
    probe = probe.replace(anchor,
        '      } else if (op == "dump_kv") {\n'
        '        const auto dir = std::filesystem::path(FLAGS_output_dir) /\n'
        '            request.at("directory").get<std::string>();\n'
        '        require(dir.parent_path() == std::filesystem::path(FLAGS_output_dir), "dump path");\n'
        '        require(std::filesystem::create_directory(dir), "preserve prior KV dump");\n'
        '        require(siliconbench_diagnostic_dump_logical_kv(&control,\n'
        '            request.at("seq_id").get<int32_t>(), dir.c_str()), "dump actual logical KV");\n'
        '        result["directory"] = dir.string();\n' + anchor)
    new_probe = args.output / 'probe.cpp'
    new_probe.write_text(probe)
    entries = json.loads((sdk / 'cmake-out/compile_commands.json').read_text())
    entry, = [e for e in entries if e['file'] == str(original_cpp)]
    cache_argv = shlex.split(entry['command'])
    cache_argv[cache_argv.index('-o') + 1] = str(args.output / 'cuda_kv_cache.cpp.o')
    cache_argv[cache_argv.index('-c') + 1] = str(copy_cpp)
    flags = (parent / 'probe-build/CMakeFiles/rewind_kv_probe.dir/flags.make').read_text()
    probe_argv = ['/usr/bin/c++']
    for label in ('CXX_DEFINES', 'CXX_INCLUDES', 'CXX_FLAGS'):
        line, = [s for s in flags.splitlines() if s.startswith(label + ' = ')]
        probe_argv += shlex.split(line.split(' = ', 1)[1])
    probe_argv += ['-c', str(new_probe), '-o', str(args.output / 'probe.cpp.o')]
    new_archive = args.output / archive.name
    shutil.copy2(archive, new_archive)
    link = shlex.split((parent / 'probe-build/CMakeFiles/rewind_kv_probe.dir/link.txt').read_text())
    for i, word in enumerate(link):
        if word == str(archive): link[i] = str(new_archive)
        elif word.endswith('mini_probe.cpp.o'): link[i] = str(args.output / 'probe.cpp.o')
    # CMake dependency-file paths are diagnostic build paths, not required here.
    link = [s for s in link if not s.startswith('-Wl,--dependency-file=')]
    link[link.index('-o') + 1] = str(args.output / 'cache_bytes_probe')
    commands = [cache_argv, probe_argv,
                ['ar', 'r', str(new_archive), str(args.output / 'cuda_kv_cache.cpp.o')], link]
    with (args.output / 'build.log').open('x') as log:
        for argv in commands:
            log.write(json.dumps(argv) + '\n'); log.flush()
            subprocess.run(argv, cwd=parent / 'probe-build', stdout=log,
                           stderr=subprocess.STDOUT, check=True, timeout=600)
    assert before == {str(f): digest(f) for f in controls}
    proof = dict(original_controls=before, diagnostic_only=True,
        original_sources_and_binaries_preserved=True, no_fields_or_virtual_changes=True,
        no_inference_kernel_changes=True, probe=str(args.output / 'cache_bytes_probe'),
        probe_SHA256=digest(args.output / 'cache_bytes_probe'),
        diagnostic_CPP_SHA256=digest(copy_cpp), probe_CPP_SHA256=digest(new_probe))
    (args.output / 'proof.json').write_text(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof))


if __name__ == '__main__': main()
