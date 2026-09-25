# Arm operator support documentation

`generate_op_support.py` generates operator support tables for VGF,
Ethos-U55, and Ethos-U85 from backend test definitions. It also checks
static test coverage against backend support registries.

Run the commands below from the ExecuTorch repository root, with the
ExecuTorch and Arm development dependencies installed.

## Generate support tables

Generate the table for the backend you changed:

```bash
python backends/arm/scripts/docgen/generate_op_support.py --backend vgf
python backends/arm/scripts/docgen/generate_op_support.py --backend u55
python backends/arm/scripts/docgen/generate_op_support.py --backend u85
```

The generated files are:

- VGF: `docs/source/backends/arm-vgf/VGF_op_support.md`
- Ethos-U55: `docs/source/backends/arm-ethos-u/U55_op_support.md`
- Ethos-U85: `docs/source/backends/arm-ethos-u/U85_op_support.md`

Review and commit the generated changes alongside the source changes.
Do not edit the generated tables manually.

## Check operator coverage

Use `--check` to validate that backend-supported operator/profile pairs
have the expected static test evidence:

```bash
python backends/arm/scripts/docgen/generate_op_support.py --backend vgf --check
python backends/arm/scripts/docgen/generate_op_support.py --backend u55 --check
python backends/arm/scripts/docgen/generate_op_support.py --backend u85 --check
```

This checks test definitions; it does not execute the tests, regenerate
the tables, or check whether committed tables are up to date.

If coverage is missing, inspect the reported operator and profile,
then add or correct the relevant backend test. Run the coverage check
again and regenerate the table.

## Generate a detailed report

Use `--debug` to include exported operators and associated tests.
Add `--html` to produce an HTML version alongside the Markdown report:

```bash
python backends/arm/scripts/docgen/generate_op_support.py \
  --backend u85 \
  --debug \
  --html \
  --output /tmp/U85_op_support_debug.md
```

Replace `u85` with `vgf` or `u55` as needed. This example writes diagnostic
reports to `/tmp` without replacing the committed support table.

## GitHub Actions checks

Pull request CI checks operator coverage and regenerates all three
support tables to compare them with the committed versions.

If CI reports outdated documentation, regenerate the tables:

```bash
for backend in vgf u55 u85; do
  python backends/arm/scripts/docgen/generate_op_support.py \
    --backend "$backend" || break
done
```

Review the changes:

```bash
git diff -- \
  docs/source/backends/arm-vgf/VGF_op_support.md \
  docs/source/backends/arm-ethos-u/U55_op_support.md \
  docs/source/backends/arm-ethos-u/U85_op_support.md
```

Include the updated tables in your commit and push the changes.