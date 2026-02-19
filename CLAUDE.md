# ExecuTorch

## Skills
- `/setup` - Set up environment
- `/export` - Export model to .pte
- `/building` - Build runners or C++ libs
- `/profile` - Profile execution
- `/cortex-m` - Build, test, or develop the Cortex-M backend

Reference docs in `.claude/`: backends, runtime-api, quantization, llm-export, faq, tokenizers

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
