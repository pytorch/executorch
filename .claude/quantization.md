# Quantization

Docs: https://docs.pytorch.org/ao/main/pt2e_quantization/index.html

## Backend quantizers
| Backend | Quantizer |
|---------|-----------|
| XNNPACK | `XNNPACKQuantizer` |
| Qualcomm | `QnnQuantizer` |
| CoreML | `CoreMLQuantizer` |

## LLM modes
See `examples/models/llama/source_transformation/quantize.py`: `int8`, `8da4w`, `4w`
