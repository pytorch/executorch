"""Patch an isolated candidate checkout; refuses to touch an existing baseline."""
import argparse,hashlib,json,os,shutil
from pathlib import Path

def once(s,a,b):
    assert s.count(a)==1,(a[:120],s.count(a));return s.replace(a,b)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def patch_native(p):
    n='extension/llm/cache/cache.h';s=(p/n).read_text()
    s=once(s,'  bool independent_sequence_capacity = false;',
        '  bool independent_sequence_capacity = false;\n  // ABI4: window-only GPU ring, shared packed staging, host rewind archive.\n  bool split_ring_history = false;')
    (p/n).write_text(s)
    n='backends/cuda/batching/cuda_executor.cpp';s=(p/n).read_text()
    s=s.replace('sequence_info_abi == 2 || sequence_info_abi == 3','sequence_info_abi == 2 || sequence_info_abi == 3 || sequence_info_abi == 4')
    s=once(s,'  cfg.independent_sequence_capacity = sequence_info_abi == 3;',
        '  cfg.independent_sequence_capacity = sequence_info_abi == 3 || sequence_info_abi == 4;\n  cfg.split_ring_history = sequence_info_abi == 4;')
    (p/n).write_text(s)
    n='backends/cuda/runtime/cuda_kv_pool.h';s=(p/n).read_text()
    s=once(s,'  runtime::Error mark_step_done();','  runtime::Error mark_step_done();\n  // Fence and trim only unused pages; live graph/KV allocations stay resident.\n  runtime::Error trim_idle_memory(uint64_t unused_allowance);')
    (p/n).write_text(s)
    n='backends/cuda/runtime/cuda_kv_pool.cpp';s=(p/n).read_text()
    idx=s.index('void CudaKVPool::trim_sequence_memory_pool(')
    code='''runtime::Error CudaKVPool::trim_idle_memory(uint64_t unused_allowance) {
  ET_CHECK_OK_OR_RETURN_ERROR(synchronize());
  const auto device = CUDAGuard::create(device_);
  ET_CHECK_OK_OR_RETURN_ERROR(device.error());
  auto trim = [&](cudaMemPool_t p) -> runtime::Error {
    if (!p) return runtime::Error::Ok;
    uint64_t used = 0;
    ET_CHECK_OR_RETURN_ERROR(cudaMemPoolGetAttribute(p, cudaMemPoolAttrUsedMemCurrent, &used) == cudaSuccess,
        Internal, "cannot read idle pool usage");
    const uint64_t keep = used > UINT64_MAX - unused_allowance ? UINT64_MAX : used + unused_allowance;
    ET_CHECK_OR_RETURN_ERROR(cudaMemPoolTrimTo(p, keep) == cudaSuccess, Internal, "cannot trim idle pool");
    return runtime::Error::Ok;
  };
  ET_CHECK_OK_OR_RETURN_ERROR(trim(sequence_memory_pool_));
  return trim(::executorch::extension::cuda::CudaAllocator::pool_for_device(device_));
}

'''
    s=s[:idx]+code+s[idx:];(p/n).write_text(s)
    n='backends/cuda/runtime/cuda_kv_cache.cpp';s=(p/n).read_text()
    # Restrict changes to the batched sequence class, leaving other layouts alone.
    a=s.index('class CudaBatchedSequenceCache');b=s.index('std::shared_ptr<cache::Cache> make_cuda_batched_sequence_kv_cache')
    before,t,after=s[:a],s[a:b],s[b:]
    t=once(t,'side_specs(cfg),','side_specs(geometry, cfg),')
    t=once(t,'      layer_rows_.push_back(rows_of(layer, cfg));\n    }',
        '      layer_rows_.push_back(rows_of(layer, cfg));\n    }\n    error_ = initialize_ring_journal();')
    t=once(t,'  ~CudaBatchedSequenceCache() override {','  ~CudaBatchedSequenceCache() override {\n    destroy_ring_journal();')
    # Cache clear already fences all streams before dropping owner bytes.
    t=once(t,'    seqs_.clear();','    clear_ring_journal();\n    seqs_.clear();')
    t=once(t,'    const int length = source->second.plan.length();',
        '    if (finish_ring_journal() != Error::Ok) return std::nullopt;\n    const int length = source->second.plan.length();')
    t=once(t,'    if (copy_column(source->second.column, seqs_.at(*id).column, keep) !=',
        '    if (cfg_.split_ring_history) ring_archives_[*id] = ring_archives_.at(src);\n    if ((keep != length && restore_ring_history(*id, seqs_.at(*id).column, keep) != Error::Ok) ||\n        copy_column(source->second.column, seqs_.at(*id).column, keep, keep == length) !=')
    t=once(t,'    return seqs_.erase(seq_id) > 0;',
        '    if (finish_ring_journal() != Error::Ok) return false;\n    ring_archives_.erase(seq_id);\n    return seqs_.erase(seq_id) > 0;')
    t=once(t,'    return it != seqs_.end() && it->second.plan.rewind(position);',
        '''    if (it == seqs_.end() || inflight_ || !it->second.plan.can_rewind(position)) return false;
    if (cfg_.split_ring_history) {
      const auto guard = CUDAGuard::create(pool_.device());
      if (!guard.ok() || pool_.follow_previous_step(pool_.stream()) != Error::Ok ||
          restore_ring_history(seq_id, it->second.column, position) != Error::Ok ||
          pool_.mark_step_done() != Error::Ok) return false;
    }
    return it->second.plan.rewind(position);''')
    # Prepare follows previous deferred D2H before overwriting shared staging.
    pos=t.index('  Error prepare_step(int64_t write_length, cudaStream_t stream) override {')
    start,end=t[:pos],t[pos:]
    anchor='  Error prepare_step(int64_t write_length, cudaStream_t stream) override {\n    std::lock_guard<std::mutex> guard(mutex_);\n    ET_CHECK_OK_OR_RETURN_ERROR(error_);'
    end=once(end,anchor,anchor+'\n    ET_CHECK_OK_OR_RETURN_ERROR(finish_ring_journal());')
    end=once(end,'    // Rows every column must hold once this step lands, and those it holds now.',
        '''    bool prefill = false;
    for (const auto& run : runs) prefill |= run.length > 1;
    if (allocator_spare_bytes() >= 0 && previous_step_prefill_ && !prefill) {
      ET_CHECK_OK_OR_RETURN_ERROR(pool_.trim_idle_memory(allocator_spare_bytes()));
      ++allocator_phase_trims_;
    }
    previous_step_prefill_ = prefill;
    ring_runs_.clear();
    ring_step_tokens_ = tokens;
    // Rows every column must hold once this step lands, and those it holds now.''')
    end=once(end,'      next[run.seq_id] = end;',
        '      next[run.seq_id] = end;\n      if (cfg_.split_ring_history) ring_runs_.push_back({run.seq_id, run.start, run.length, end - run.length});')
    # Commit copy ends before the pool's last-step fence, with no extra host wait.
    pos2=end.index('  Error commit_step(int64_t write_length) override {');u,v=end[:pos2],end[pos2:]
    v=once(v,'    ET_CHECK_OK_OR_RETURN_ERROR(pool_.mark_step_done());\n    inflight_ = false;',
        '    ET_CHECK_OK_OR_RETURN_ERROR(copy_ring_journal());\n    ET_CHECK_OK_OR_RETURN_ERROR(pool_.mark_step_done());\n    inflight_ = false;')
    t=start+u+v
    t=once(t,'    return ensure_columns(true, spare_columns, pool_.stream());',
        '''    ET_CHECK_OK_OR_RETURN_ERROR(finish_ring_journal());
    ET_CHECK_OK_OR_RETURN_ERROR(ensure_columns(true, spare_columns, pool_.stream()));
    if (allocator_spare_bytes() >= 0) {
      ET_CHECK_OK_OR_RETURN_ERROR(pool_.trim_idle_memory(allocator_spare_bytes()));
      ++allocator_phase_trims_;
    }
    return Error::Ok;''')
    t=once(t,'side_specs(const cache::CacheConfig& cfg)',
        'side_specs(const cache::CacheGeometry& geometry, const cache::CacheConfig& cfg)')
    t=once(t,'    return specs;',
        '''    if (cfg.split_ring_history) {
      int rings = 0; int heads = 0; int dim = 0;
      for (const auto& layer : geometry.layers) if (is_ring(layer)) {
        ++rings; heads = layer.n_kv_heads; dim = layer.head_dim;
      }
      specs.push_back({"__et_offgraph_kv_ring_stage", slimc10::ScalarType::BFloat16,
          {rings, 2, *cfg.max_write, heads, dim}});
    }
    return specs;''')
    t=once(t,'static_cast<int64_t>(layer.policy.window) + *cfg.max_write - 1);',
        'static_cast<int64_t>(layer.policy.window) + (cfg.split_ring_history ? 0 : *cfg.max_write - 1));')
    # Full-cache clone remains original; the ring is restored from the undo archive.
    t=once(t,'      if (is_ring(layer)) {','      if (is_ring(layer)) {\n        if (cfg_.split_ring_history && !copy_current_ring) continue;')
    t=once(t,'Error copy_column(int from, int to, int keep)', 'Error copy_column(int from, int to, int keep, bool copy_current_ring = false)')
    insertion='''  bool previous_step_prefill_ = false;
  uint64_t allocator_phase_trims_ = 0;
  static int64_t allocator_spare_bytes() {
    static const int64_t value = []() -> int64_t {
      const char* text = std::getenv("ET_CUDA_IDLE_SPARE_BYTES");
      if (!text || !*text) return -1;
      char* end = nullptr; const auto v = std::strtoll(text, &end, 10);
      return end && !*end && v >= 0 && v <= (1LL << 40) ? v : -1;
    }();
    return value;
  }
#include <executorch/backends/cuda/runtime/ring_host_journal.inc>
'''
    t=once(t,'  mutable std::mutex mutex_;',insertion+'\n  mutable std::mutex mutex_;')
    after=once(after,'  return std::make_shared<CudaBatchedSequenceCache>(',
        '''  if (cfg.split_ring_history && (!cfg.independent_sequence_capacity ||
      geometry.dflash_block_size || storage_dtype != slimc10::ScalarType::BFloat16)) {
    ET_LOG(Error, "split ring ABI4 requires solo independent BF16 KV");
    return nullptr;
  }
  return std::make_shared<CudaBatchedSequenceCache>(''')
    s=before+t+after;s=once(s,'#include <cstring>','#include <cstring>\n#include <cstdlib>')
    (p/n).write_text(s)


def patch_python(p):
    n='backends/cuda/passes/lower_offgraph_kv.py';s=(p/n).read_text()
    s=s.replace('not in (1, 2, 3)','not in (1, 2, 3, 4)').replace('independent != (manifest["version"] == 3)','independent != (manifest["version"] in (3, 4))').replace('manifest["version"] in (2, 3)','manifest["version"] in (2, 3, 4)')
    s=once(s,'    if manifest["version"] in (2, 3, 4):',
        '''    split = manifest.setdefault("split_ring_history", False)
    if not isinstance(split, bool) or split != (manifest["version"] == 4):
        raise ValueError("split ring history requires explicit manifest version4")
    if split and (not independent or manifest.get("context_window", False) or not manifest.get("causal", True)):
        raise ValueError("split ring requires independent causal solo attention")
    if manifest["version"] in (2, 3, 4):''')
    s=once(s,'            rows = ring_physical_capacity(layer["window"], self._manifest["max_write"])',
        '            rows = layer["window"] if self._manifest.get("split_ring_history", False) else ring_physical_capacity(layer["window"], self._manifest["max_write"])')
    marker='        return fn, call_args, example_args\n\n    def _ring_mask'
    inject='''        if ring and self._manifest.get("split_ring_history", False):
            from executorch.backends.cuda.triton.kernels.two_source_ring import sdpa_ring_current, ring_commit_journal
            ring_layers = [l for l in self._manifest["layers"] if l["policy"] == "ring"]
            ring_index = next(i for i,l in enumerate(ring_layers) if l is layer)
            kv = node.args[1].meta["val"]
            heads, dim = kv.shape[1], kv.shape[3]
            if any(l["window"] != layer["window"] for l in ring_layers):
                raise ValueError("split ring requires uniform ring geometry")
            stage = self._cell_buffer(exported_program, "__et_offgraph_kv_ring_stage",
                (len(ring_layers), 2, self._manifest["max_write"], heads, dim), torch.bfloat16, device)
            call_args = (*call_args, stage)
            example_args = (*example_args, stage.meta["val"])
            def split_ring_fn(q,k,v,position,ks,vs,cols,info,base,stage):
                kh = ks.view(rows, columns, heads, dim)
                vh = vs.view(rows, columns, heads, dim)
                out = sdpa_ring_current(q,kh,vh,k,v,info,window,scale)
                ring_commit_journal(kh,vh,k,v,info,window,stage,ring_index)
                return out.to(out_dtype)
            fn = split_ring_fn
        return fn, call_args, example_args

    def _ring_mask'''
    s=once(s,marker,inject);(p/n).write_text(s)
    n='examples/models/muse-glimmer/export/export_solo.py';s=(p/n).read_text()
    # Environment opt-in is intentionally local to a new export. ABI4 advertises
    # it so old native runtimes refuse to load the candidate artifact.
    s=once(s,'    if context_window:\n        manifest["context_window"] = True',
        '''    if os.environ.get("ET_CUDA_SPLIT_RING_HISTORY", "0") == "1":
        if not independent_capacity or context_window or not causal:
            raise ValueError("split ring export requires independent causal solo mode")
        manifest["version"] = 4
        manifest["split_ring_history"] = True
    if context_window:
        manifest["context_window"] = True''')
    s=once(s,'constant_methods["get_offgraph_kv_sequence_info_abi"] = 3 if independent_capacity else 2',
        'constant_methods["get_offgraph_kv_sequence_info_abi"] = 4 if os.environ.get("ET_CUDA_SPLIT_RING_HISTORY", "0") == "1" else (3 if independent_capacity else 2)')
    if '\nimport os\n' not in s:s=once(s,'import json','import json\nimport os')
    (p/n).write_text(s)
    gemma=p/'examples/models/gemma4_31b/export_batching.py'
    if gemma.exists():
        gs=gemma.read_text()
        gs=once(gs,'metadata["get_offgraph_kv_sequence_info_abi"] = 3 if independent_capacity else 2',
            'metadata["get_offgraph_kv_sequence_info_abi"] = 4 if os.environ.get("ET_CUDA_SPLIT_RING_HISTORY", "0") == "1" else (3 if independent_capacity else 2)')
        if '\nimport os\n' not in gs:gs=once(gs,'import json','import json\nimport os')
        gemma.write_text(gs)


def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--baseline',type=Path,required=True);a.add_argument('--output',type=Path,required=True)
    a.add_argument('--role',choices=['native','export','combined'],default='combined')
    args=a.parse_args();assert not args.output.exists();assert args.output.resolve()!=args.baseline.resolve()
    package=Path(__file__).parent
    baseline_files=['extension/llm/cache/cache.h','backends/cuda/batching/cuda_executor.cpp','backends/cuda/runtime/cuda_kv_cache.cpp','backends/cuda/runtime/cuda_kv_pool.cpp','backends/cuda/runtime/cuda_kv_pool.h','backends/cuda/passes/lower_offgraph_kv.py','examples/models/muse-glimmer/export/export_solo.py']
    before={n:sha(args.baseline/n) for n in baseline_files}
    shutil.copytree(args.baseline,args.output,symlinks=True,ignore=shutil.ignore_patterns('.git','cmake-out','__pycache__','.pytest_cache','*.pte','*.ptd','*.gguf','*.log','*.nsys-rep'))
    if args.role in ('native','combined'):patch_native(args.output)
    if args.role in ('export','combined'):patch_python(args.output)
    for name,dest in [('two_source_ring.py','backends/cuda/triton/kernels/two_source_ring.py'),('ring_host_journal.inc','backends/cuda/runtime/ring_host_journal.inc')]:
        (args.output/dest).parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(package/name,args.output/dest)
    if args.role in ('native','combined') and (args.output/'backends/cuda/runtime/test/test_cuda_kv_cache.cpp').exists():
        dst=args.output/'backends/cuda/runtime/test/ring_retention_native_tests.inc'
        shutil.copy2(package/'ring_retention_native_tests.inc',dst)
        test=args.output/'backends/cuda/runtime/test/test_cuda_kv_cache.cpp'
        ts=test.read_text();assert '#include "ring_retention_native_tests.inc"' not in ts
        test.write_text(ts+'\n#include "ring_retention_native_tests.inc"\n')
    assert before=={n:sha(args.baseline/n) for n in baseline_files}
    (args.output/'ring-retention-source-audit.json').write_text(json.dumps(dict(
        baseline=str(args.baseline),candidate=str(args.output),baseline_SHA256=before,
        candidate_SHA256={n:sha(args.output/n) for n in baseline_files},
        additional_candidate_SHA256={n:sha(args.output/n) for n in ["backends/cuda/triton/kernels/two_source_ring.py","backends/cuda/runtime/ring_host_journal.inc","examples/models/gemma4_31b/export_batching.py"] if (args.output/n).exists()},
        old_source_unchanged=True,ABI4_window_only_history=True,host_undo_archive_preserves_max_write_rewind=True,
        no_model_exported=True,performance_accepted=False,mini_model_qualification_pending=True),indent=2)+'\n')
if __name__=='__main__':main()
