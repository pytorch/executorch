---
name: coreai-code-writing
description: >-
  Conventions for writing code in the Core AI ExecuTorch backend
  (backends/apple/coreai): comment style (clear, concise, self-documenting) and
  how to run the test suite. Apply when editing or adding code/tests here.
---

# Core AI backend: code-writing conventions

## Comments and docstrings

Write self-documenting code first; reach for a comment only when the code cannot
speak for itself.

- **Explain WHY, not WHAT.** Well-named identifiers already say what the code
  does. Comment only non-obvious reasons: a hidden constraint, a subtle
  invariant, a workaround, or behavior that would surprise a reader.
- **Be clear and concise.** One short line is usually enough. Prefer a plain
  sentence over a paragraph. Delete a comment if removing it wouldn't confuse a
  future reader.
- **No LLMisms.** Do not use em-dashes (`—`) or ` -- ` as sentence separators;
  use periods, commas, semicolons, or parentheses. Do not use decorative dash
  banners (`# --- section ---`); a plain `# Section.` line is enough. Avoid
  markdown emphasis (`*x*`, `**x**`) and arrows (`=>`) in prose.
- **Don't restate the task or the diff.** No "added for X", "used by Y", or
  "handles the case from T123"; that belongs in the diff description and rots.
- **Keep comments true.** When you change code, update or delete any comment it
  affects rather than leaving a stale claim.

Legitimate non-prose dashes are fine: CLI flags (`--platform`,
`--min-deployment-version`) and shell/code tokens.

## Running the tests

Run the whole backend suite (both `test/` and `passes/test/`) with the helper
script, from inside the Core AI conda env:

```bash
conda run -n coreai backends/apple/coreai/run_all_tests.sh
```

Extra args are forwarded to `unittest`, e.g. `... run_all_tests.sh -v` or
`... run_all_tests.sh -k sidecar`.

**Use this script (unittest), not `pytest` directly.** pytest's default
discovery puts `backends/apple/` on `sys.path`, so `import coreai` resolves to
this backend directory and shadows the real Apple `coreai` SDK (`coreai_torch`
then fails importing `coreai.authoring`). unittest imports via the full
`executorch.backends.apple.coreai.*` path and avoids the shadowing.

The real-toolchain AOT test (`CoreAIAOTCompileTest`) is gated on
`xcrun coreai-build`; it runs on macOS with the Metal Toolchain and skips
elsewhere. The mocked AOT tests always run.
