#!/usr/bin/env bash

set -euo pipefail

script_dir=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
repo_root=$(cd "${script_dir}/../../.." && pwd)

mode="all"
quantize_scope="linear"
output_dir="${repo_root}"
tokenizer_path="${repo_root}/data/tokenizers/smollm2/tokenizer.json"
max_seq_length=64
max_context_length=64
calibration_limit=1
calibration_seq_length=64
target="ethos-u85-256"
system_config="Ethos_U85_SYS_DRAM_High"
memory_mode="Dedicated_Sram_512KB"
full_logits=0
use_kv_cache=1
static_quantize_kv_cache=0
so_library=""
ethosu_extra_flags=""

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --mode=all|w8a8|w8a16          Which export(s) to generate. Default: ${mode}
  --quantize_scope=full|linear   Arm PT2E quantization scope. Default: ${quantize_scope}
  --output_dir=DIR               Output directory. Default: ${output_dir}
                                 The repo-root default matches the quickstart.
  --tokenizer=PATH               Tokenizer JSON path. Default: ${tokenizer_path}
  --so_library=PATH              Quantized AOT ops library used to register export
                                 out variants for static KVQ.
  --max_seq_length=N             Export window size. Default: ${max_seq_length}
  --max_context_length=N         Export context size. Default: ${max_context_length}
  --calibration_limit=N          Wikitext sample count. Default: ${calibration_limit}
  --calibration_seq_length=N     Calibration token window. Default: ${calibration_seq_length}
  --target=NAME                  Ethos-U target. Default: ${target}
  --system_config=NAME           Vela system config. Default: ${system_config}
  --memory_mode=NAME             Vela memory mode. Default: ${memory_mode}
  --ethosu_extra_flags=LIST      JSON-style Hydra list of extra Vela flags, e.g.
                                 '["--arena-cache-size=1048576"]'
  --full_logits                  Export static non-KV full logits and append
                                 _full_logits to filenames.
  --use_kv_cache|--use-kv-cache  Export KV-cache model inputs. Full logits are disabled.
                                 This is the default for the quickstart.
  --static_quantize_kv_cache|--static-quantize-kv-cache
                                 Store KV cache as calibrated static int8. Requires
                                 --mode=w8a16 and --quantize_scope=linear.
  --no_kv_cache|--no-kv-cache    Export the static non-KV fallback path.
                                 Full logits are enabled by default for this path.
EOF
}

for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    --mode=*) mode="${arg#*=}" ;;
    --quantize_scope=*) quantize_scope="${arg#*=}" ;;
    --output_dir=*) output_dir="${arg#*=}" ;;
    --tokenizer=*) tokenizer_path="${arg#*=}" ;;
    --so_library=*) so_library="${arg#*=}" ;;
    --max_seq_length=*) max_seq_length="${arg#*=}" ;;
    --max_context_length=*) max_context_length="${arg#*=}" ;;
    --calibration_limit=*) calibration_limit="${arg#*=}" ;;
    --calibration_seq_length=*) calibration_seq_length="${arg#*=}" ;;
    --target=*) target="${arg#*=}" ;;
    --system_config=*) system_config="${arg#*=}" ;;
    --memory_mode=*) memory_mode="${arg#*=}" ;;
    --ethosu_extra_flags=*) ethosu_extra_flags="${arg#*=}" ;;
    --full_logits) full_logits=1; use_kv_cache=0 ;;
    --use_kv_cache|--use-kv-cache) use_kv_cache=1; full_logits=0 ;;
    --static_quantize_kv_cache|--static-quantize-kv-cache) static_quantize_kv_cache=1 ;;
    --no_kv_cache|--no-kv-cache) use_kv_cache=0; full_logits=1 ;;
    *)
      echo "Unknown option: ${arg}" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "${static_quantize_kv_cache}" -eq 1 ]]; then
  if [[ "${use_kv_cache}" -ne 1 ]]; then
    echo "Static quantized KV cache requires --use-kv-cache." >&2
    exit 1
  fi
  if [[ "${mode}" != "w8a16" ]]; then
    echo "Static quantized KV cache requires --mode=w8a16." >&2
    exit 1
  fi
  if [[ "${quantize_scope}" != "linear" ]]; then
    echo "Static quantized KV cache requires --quantize_scope=linear." >&2
    exit 1
  fi
fi

mkdir -p "${output_dir}"

run_export() {
  local pt2e_quantize="$1"
  local output_name="$2"

  echo "[export] output_name=${output_name}"
  echo "[export] backend.ethosu.extra_flags=${ethosu_extra_flags:-[] }"

  local -a cmd=(
    python -m extension.llm.export.export_llm
    base.model_class=smollm2
    base.params=examples/models/smollm2/135M_config.json
    base.tokenizer_path="${tokenizer_path}"
    export.output_dir="${output_dir}"
    export.output_name="${output_name}"
    export.max_seq_length="${max_seq_length}"
    export.max_context_length="${max_context_length}"
    quantization.pt2e_quantize="${pt2e_quantize}"
    quantization.quantize_scope="${quantize_scope}"
    quantization.calibration_tasks="[wikitext]"
    quantization.calibration_limit="${calibration_limit}"
    quantization.calibration_seq_length="${calibration_seq_length}"
    backend.ethosu.enabled=True
    backend.ethosu.target="${target}"
    backend.ethosu.system_config="${system_config}"
    backend.ethosu.memory_mode="${memory_mode}"
    model.use_kv_cache=$( [[ "${use_kv_cache}" -eq 1 ]] && echo True || echo False )
    model.enable_dynamic_shape=False
    debug.verbose=True
    debug.generate_full_logits=$( [[ "${full_logits}" -eq 1 ]] && echo True || echo False )
  )
  if [[ "${static_quantize_kv_cache}" -eq 1 ]]; then
    cmd+=("model.static_quantize_kv_cache=True")
  fi
  if [[ -n "${so_library}" ]]; then
    cmd+=("export.so_library=${so_library}")
  fi
  if [[ -n "${ethosu_extra_flags}" ]]; then
    cmd+=("backend.ethosu.extra_flags=${ethosu_extra_flags}")
  fi

  "${cmd[@]}"
}

output_name_for() {
  local stem="$1"
  if [[ "${static_quantize_kv_cache}" -eq 1 ]]; then
    stem="smollm2_ethosu_static_kvq_seq${max_seq_length}_${stem}"
  elif [[ "${use_kv_cache}" -eq 1 ]]; then
    stem="smollm2_ethosu_kv_seq${max_seq_length}_${stem}"
  else
    stem="smollm2_ethosu_seq${max_seq_length}_${stem}"
  fi
  if [[ "${full_logits}" -eq 1 ]]; then
    printf '%s_full_logits.pte' "${stem}"
  else
    printf '%s.pte' "${stem}"
  fi
}

cd "${repo_root}"

case "${mode}" in
  all)
    run_export ethosu_8a8w "$(output_name_for w8a8_wikitext)"
    run_export ethosu_16a8w "$(output_name_for w8a16_wikitext)"
    ;;
  w8a8)
    run_export ethosu_8a8w "$(output_name_for w8a8_wikitext)"
    ;;
  w8a16)
    run_export ethosu_16a8w "$(output_name_for w8a16_wikitext)"
    ;;
  *)
    echo "Unsupported mode: ${mode}" >&2
    exit 1
    ;;
esac
