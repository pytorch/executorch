# Git Hooks

This directory contains Git hooks for the ExecuTorch repository.

## Pre-commit Hook

The pre-commit hook automatically updates the PyTorch commit pin in `.ci/docker/ci_commit_pins/pytorch.txt` whenever `torch_pin.py` is modified.

### How It Works

1. When you commit changes to `torch_pin.py`, the hook detects the change
2. It parses the `NIGHTLY_VERSION` field (e.g., `dev20251004`)
3. Converts it to a date string (e.g., `2025-10-04`)
4. Fetches the corresponding commit hash from the PyTorch nightly branch at https://github.com/pytorch/pytorch/tree/nightly
5. Updates `.ci/docker/ci_commit_pins/pytorch.txt` with the new commit hash
6. Automatically stages the updated file for commit

### Installation

To install the Git hooks, run:

```bash
.githooks/install.sh
```

This will copy the pre-commit hook to `.git/hooks/` and make it executable.

### Manual Usage

You can also run the update script manually at any time:

```bash
python .github/scripts/update_pytorch_pin.py
```

### Uninstalling

To remove the pre-commit hook:

```bash
rm .git/hooks/pre-commit
```

## Troubleshooting

If the hook fails during a commit:

1. Check that Python 3 is available in your PATH
2. Ensure you have internet connectivity to fetch commits from GitHub
3. Verify that the `NIGHTLY_VERSION` in `torch_pin.py` is in the correct format (`devYYYYMMDD`)
4. Make sure the corresponding nightly release exists in the PyTorch nightly branch

You can run the script manually to see detailed error messages:

```bash
python .github/scripts/update_pytorch_pin.py
```
