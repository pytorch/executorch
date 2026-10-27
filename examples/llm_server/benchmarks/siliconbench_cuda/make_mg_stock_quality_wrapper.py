"""Generate an isolated MG correctness launcher using stock server switches.

Model and tokenizer directories are read-only inputs. Runtime caches and
temporary files belong to a new task directory. This does not modify either
framework, model weights, templates, or previous experiment records.
"""
import argparse
from pathlib import Path
import shlex
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['model-path', 'tokenizer-path', 'vllm-bin', 'sglang-python', 'task-root', 'output']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--reuse-runtime-cache', action='store_true',
                   help='Reuse existing task-owned compiler caches without changing framework code')
    a = p.parse_args()
    assert not a.output.exists()
    assert a.task_root.is_dir() if a.reuse_runtime_cache else not a.task_root.exists()
    assert (a.model_path/'download-audit.json').is_file()
    assert (a.tokenizer_path/'chat_template.jinja').is_file()
    assert a.vllm_bin.is_file() and a.sglang_python.is_file()
    for name in ['triton', 'inductor', 'flashinfer', 'hf', 'xdg', 'cuda', 'tmp']:
        (a.task_root/name).mkdir(parents=True, exist_ok=a.reuse_runtime_cache)
    assignments = '\n'.join(name+'='+shlex.quote(str(value)) for name,value in [
        ('model',a.model_path),('tokenizer',a.tokenizer_path),('vl',a.vllm_bin),
        ('sg',a.sglang_python),('task',a.task_root)])
    script = r'''#!/usr/bin/env bash
set -euo pipefail
backend=${1:?}
ASSIGNMENTS
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 VLLM_PLUGINS=""
unset PYTHONPATH SB_ET_PREFILL_TRACE SB_PREFILL_A16 SB_PREFILL_TRACE_DIR GGML_CUDA_CUBLAS_COMPUTE_TYPE
export CUDA_HOME=/usr/local/cuda TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
export TRITON_CACHE_DIR="$task/triton" TORCHINDUCTOR_CACHE_DIR="$task/inductor"
export FLASHINFER_WORKSPACE_BASE="$task/flashinfer" HF_HOME="$task/hf" XDG_CACHE_HOME="$task/xdg"
export CUDA_CACHE_PATH="$task/cuda" TMPDIR="$task/tmp"
port=${SB_PORT:-19082}
case "${SB_PREFIX_CACHE:?}" in
  on) sg_cache_args=(); vl_cache_args=(--enable-prefix-caching) ;;
  off) sg_cache_args=(--disable-radix-cache); vl_cache_args=(--no-enable-prefix-caching) ;;
  *) exit 2 ;;
esac
case "$backend" in
  sglang)
    exec "$sg" -m sglang.launch_server "${sg_cache_args[@]}" \
      --model-path "$model" --tokenizer-path "$tokenizer" --served-model-name muse-glimmer-30b \
      --language-model-only --host 127.0.0.1 --port "$port" --context-length 131072 \
      --max-running-requests 16 --mem-fraction-static 0.85 \
      --quantization compressed-tensors --dtype bfloat16 --kv-cache-dtype bf16 \
      --chunked-prefill-size 8192 --max-prefill-tokens 8192 --cuda-graph-max-bs-decode 16 \
      --cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 512 \
      --default-chat-template-kwargs '{"current_date":"2026-10-21","reasoning_strength":"high","enable_thinking":false}' \
      --reasoning-parser muse --tool-call-parser muse
    ;;
  vllm)
    export VLLM_USE_V2_MODEL_RUNNER=1
    exec "$vl" serve "$model" --served-model-name muse-glimmer-30b --tokenizer "$tokenizer" \
      --language-model-only --quantization compressed-tensors --dtype bfloat16 --kv-cache-dtype bfloat16 \
      --host 127.0.0.1 --port "$port" --max-model-len 131072 --max-num-seqs 16 \
      --gpu-memory-utilization 0.85 --max-num-batched-tokens 8192 "${vl_cache_args[@]}" \
      --generation-config auto --chat-template-content-format string \
      --reasoning-parser muse_glimmer \
      --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[1,2,4,8,16]}'
    ;;
  *) exit 2 ;;
esac
'''.replace('ASSIGNMENTS', assignments)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(script)
    subprocess.run(['bash','-n',str(a.output)],check=True)


if __name__ == '__main__':
    main()
