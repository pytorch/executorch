# Common Errors

## Error Codes
Error codes defined in `runtime/core/error.h`.

| Code | Name | Common Cause |
|------|------|--------------|
| 0x10 | InvalidArgument | Input shape mismatch - inputs don't match export shapes. Use dynamic shapes if needed. |
| 0x14 | OperatorMissing | Selective build missing operator. Regenerate `et_operator_library` from current model. |
| 0x20 | NotFound | Missing backend. Link with `--whole-archive`: `-Wl,--whole-archive libxnnpack_backend.a -Wl,--no-whole-archive` |

## Export Issues

**Missing out variants**: Custom ops need ExecuTorch implementation. See `kernel-library-custom-aten-kernel.md`.

**RuntimeError: convert function not implemented**: Unsupported operator. File GitHub issue.

## Runtime Issues

**Slow inference**:
1. Build with `-DCMAKE_BUILD_TYPE=Release`
2. Ensure model is delegated (use `XnnpackPartitioner`)
3. Set thread count: `threadpool::get_threadpool()->_unsafe_reset_threadpool(num_threads)`

**Numerical accuracy**: Use devtools to debug. See `/profile` skill.

**Error setting input 0x10**: Input shape mismatch. Specify dynamic shapes at export.

**Duplicate kernel registration abort**: Multiple `gen_operators_lib` linked. Use only one per target.

## Installation

**Missing python-dev**: `sudo apt install python<version>-dev`

**Missing pytorch_tokenizers**: `pip install -e ./extension/llm/tokenizers/`
