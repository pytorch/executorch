The `bundled_program_schema.fbs` file is for serializing bundled program. It
bundles the ExecuTorch program, several sets of inputs and referenced outputs,
and other useful info together for verifying the correctness of ExecuTorch program.

## Rules to ensure forward/backward compatibility
Please check the rules in [here](../../../schema/README.md) for more info.


## Regenerating generated code

Schema changes require regenerating the Python bindings in
`devtools/bundled_program/serialize/generated` and committing the updated files. From the repo root:

```sh
python devtools/bundled_program/serialize/generate_bundled_program.py
```