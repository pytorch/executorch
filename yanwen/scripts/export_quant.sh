#!/usr/bin/env bash
# Export one quantized Llama 3.1 8B Vulkan .pte via the export_llm CLI (Plan A).
# Storage (texture vs buffer) is chosen by ET_VK_FORCE_BUFFER — SAME branch, SAME
# command, no op_registry edit. PTE lands in pte_out with _texture/_buffer naming.
#
# Usage:  export_quant.sh <qmode> <group_size> <texture|buffer>
#   4w     128  texture
#   8da4w  128  buffer
#   torchao:8da8w  0  buffer     # int8 per-channel (dq8ca_q8csw)
#
# Run from the quant-dev worktree root with the editable venv activated.
set -euo pipefail

QMODE="${1:?qmode}"; GROUP="${2:?group_size}"; STORAGE="${3:?texture|buffer}"
CKPT=/local/yanwen.xu/models/llama3_1_8b/original
PTE_OUT=/local/yanwen.xu/workspace/pte_out

# Friendly basename: strip "torchao:" prefix for the filename.
TAG="${QMODE#torchao:}"
NAME="llama3_1_8b_${TAG}_${STORAGE}.pte"

export ET_VK_FORCE_BUFFER=""
[ "$STORAGE" = "buffer" ] && export ET_VK_FORCE_BUFFER=1
echo "[export] qmode=$QMODE group=$GROUP storage=$STORAGE (ET_VK_FORCE_BUFFER='${ET_VK_FORCE_BUFFER}') -> $NAME"

# export.output_dir is NOT honored; PTE lands in CWD. Run from a tmp dir, then mv.
WORK=$(mktemp -d)
cd "$WORK"
python -m executorch.extension.llm.export.export_llm \
  base.model_class=llama3 \
  base.checkpoint=$CKPT/consolidated.00.pth \
  base.params=$CKPT/params.json \
  base.metadata="'{\"get_bos_id\":128000,\"get_eos_ids\":[128009,128001]}'" \
  model.use_kv_cache=True \
  model.use_sdpa_with_kv_cache=True \
  model.dtype_override=fp32 \
  quantization.qmode="$QMODE" \
  quantization.group_size=$GROUP \
  backend.vulkan.enabled=True \
  backend.vulkan.force_fp16=True \
  export.max_seq_length=2048 \
  export.max_context_length=2048 \
  export.output_name="$NAME"

mkdir -p "$PTE_OUT"
mv -f "$WORK/$NAME" "$PTE_OUT/$NAME"
cd /; rm -rf "$WORK"
echo "[export] DONE -> $PTE_OUT/$NAME ($(du -h "$PTE_OUT/$NAME" | cut -f1))"
