The `bundled_program_schema.fbs` file is for serializing bundled program. It
bundles the ExecuTorch program, several sets of inputs and referenced outputs,
and other useful info together for verifying the correctness of ExecuTorch program.

Optional inputs can contain Python `None`, encoded using FlatBuffers' existing
`NONE` union tag. Their positions are preserved, including between tensor inputs.
The exported method must expect `None` at that position; this does not make a
specialized input dynamically switch between `None` and a tensor. Outputs remain
tensor-only.

This does not change the wire format or existing tensor/scalar bundles. Older
BundleIO runtimes reject bundles containing `NONE` inputs; rebuild the runtime
with `NONE` input support before using them.

## Rules to ensure forward/backward compatibility
Please check the rules in [here](../../../schema/README.md) for more info.


## Regenerating generated code

Schema changes require regenerating the Python bindings in
`devtools/bundled_program/serialize/generated` and committing the updated files. From the repo root:

```sh
python devtools/bundled_program/serialize/generate_bundled_program.py
```
