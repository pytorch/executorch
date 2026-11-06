# LFM3 on Qualcomm HTP

Static LFM3 models for the QNN static LLM flow (`examples/qualcomm/oss_scripts/llama`):

| `--decoder_model` | Hugging Face | Modality |
|---|---|---|
| `lfm3-1_7b` | LiquidAI/LFM3-1.7B | text |
| `lfm3-1_7b_base` | LiquidAI/LFM3-1.7B-Base | text (base) |

Short-convolution layers carry their rolling state as graph IO after the V caches; the runners'
`ConvStateManager` keeps the last `conv_L_cache - 1` valid rows between steps. Lookahead decoding,
attention sink and sharding are not supported for these models.

## Text
```
python examples/qualcomm/oss_scripts/llama/llama.py -b build-android -s ${SERIAL_NUM} -m SM8750 \
  --decoder_model lfm3-1_7b --model_mode hybrid --prefill_ar_len 128 --max_seq_len 1024 \
  --temperature 0 --calib_tasks wikitext --calib_limit 1 \
  --prompt "What is the capital of France? Answer in one sentence."
```

