## Deprecated Models

This readme documents deprecated models that remain compatible with versions of the ExecuTorch runtime.

ModuleLinear-no-constant-segment.pte
- This file contains constants stored in the constant_buffer, which was deprecated in D61996249, [#5096](https://github.com/pytorch/executorch/pull/5096) on 2024-09-06. Now, constants are stored in a separate segment.
- This .pte file was generated internally using hg commit hash rFBS5e49dc0319b1d2d9969bbcef92857ab76a899c34, with command:
    ```
    buck2 build fbcode//executorch/test/models:exported_programs[ModuleLinear-no-constant-segment.pte] --show-output
    ```
- In OSS, the same .pte file can be generated with https://github.com/pytorch/executorch/commit/cea5abbcdded, via:
    ```
    python -m test.models.export_program --modules "ModuleLinear" --outdir .
    ```
