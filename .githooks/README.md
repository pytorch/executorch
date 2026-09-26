# Git Hooks

This directory contains Git hooks for the ExecuTorch repository. It is used as
`core.hooksPath`, so git looks here instead of `.git/hooks/`.

## Hooks

### pre-commit

Runs on every commit:

1. **torch_pin sync** — when `torch_pin.py` is staged, updates the PyTorch commit
   pin in `.ci/docker/ci_commit_pins/pytorch.txt` and syncs grafted c10 files.
2. **lintrunner** — runs `lintrunner -a --revision HEAD^ --skip MYPY` on changed
   files. Auto-fixes formatting and blocks on lint errors. Soft-fails if lintrunner
   is not installed. Runs `lintrunner init` automatically when `.lintrunner.toml`
   changes.

### pre-push

Delegates to `.git/hooks/pre-push` if one exists. This allows backend-specific
pre-push hooks (e.g., ARM's license and commit message checks) to work alongside
the repo-wide hooks.

## Installation

Hooks are installed automatically by `./install_executorch.sh`.

To install manually:

```bash
git config core.hooksPath .githooks
```

### ARM backend pre-push

ARM contributors should additionally install the ARM-specific pre-push hook:

```bash
cp backends/arm/scripts/pre-push .git/hooks/
```

## Bypassing

To skip hooks for a single commit or push:

```bash
git commit --no-verify
git push --no-verify
```

## Troubleshooting

If the torch_pin hook fails:

1. Check that Python 3 is available in your PATH
2. Ensure you have internet connectivity to fetch commits from GitHub
3. Verify that the `NIGHTLY_VERSION` in `torch_pin.py` is in the correct format (`devYYYYMMDD`)

If lintrunner fails:

1. Run `lintrunner init` to install linter tools
2. Check that your virtual environment is activated
