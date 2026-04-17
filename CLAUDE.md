# ExecuTorch

## Skills
- `/setup` - Set up environment
- `/export` - Export model to .pte
- `/building` - Build runners or C++ libs
- `/profile` - Profile execution
- `/cortex-m` - Build, test, or develop the Cortex-M backend
- `/binary-size` - Analyze and reduce binary size
- `/qualcomm` - Build, test, or develop the QNN (Qualcomm AI Engine Direct) backend
- `/executorch-kb` - Search tribal knowledge base (known issues, quant recipes, debugging guides)

Reference docs in `.claude/`: backends, runtime-api, quantization, llm-export, faq, tokenizers
Tribal knowledge wiki in `.wiki/`: synthesized from 2,200+ GitHub issues

For error messages, SoC compatibility questions, quantization recipes, and backend-specific debugging, consult the `/executorch-kb` skill (or read `.wiki/index.md` and navigate to the relevant article). The wiki contains tribal knowledge synthesized from 2,200+ GitHub issues with source citations. For build, test, profile, setup, or general API questions, use the dedicated skills above instead — the wiki is scoped to debugging and compatibility knowledge.

## Quick Reference

**Install Python package:**
```bash
./install_executorch.sh        # first time (or .bat on Windows)
pip install -e . --no-build-isolation  # subsequent installs
```

**Build C++ libraries:** see `CMakeLists.txt`; for LLM/ASR runners use `Makefile` and `CMakePresets.json`

**Run tests:** `pytest -n auto` (Python), `ctest --output-on-failure` (C++)

**Lint:** `lintrunner init && lintrunner -a`

Details: [docs/source/using-executorch-building-from-source.md](docs/source/using-executorch-building-from-source.md)

## Long-running commands

ExecuTorch model exports and large builds (CMake configure+build of LLM runners, AOT lowering, NeMo restore, big HF downloads) can hang silently and may not surface an exit code through pipes like `tail`. For those long jobs only, poll progress every ~120s — check the process state (`ps`, `py-spy dump`), output file growth, and network/file activity — rather than waiting indefinitely on the original Bash invocation. Avoid wrapping with `| tail` for long jobs since it buffers and hides progress; tee to a log file or run unwrapped. Normal short commands don't need this — run them directly and trust the exit code.

## Naming

- Use "executorch" (lowercase) or "ExecuTorch" (camel case)
- Never "ExecutorTorch"
- "ET" only when space-constrained (unofficial)

## Commits

- Only commit when explicitly asked
- No bullet lists of changes; explain review order for large PRs, or omit for small ones
- Disclose PR was authored with Claude

## Code Style

- Minimal comments; code should be self-documenting
- Comments only for non-obvious global context
- No trivial (1-2 LOC) single-use helpers unless significantly improving readability
- Explicit state management; no dynamic `setattr`/`getattr` patterns
- Match existing style and architecture
- Assume reader knows ExecuTorch/PyTorch basics

**When uncertain: choose simpler, more concise.**
