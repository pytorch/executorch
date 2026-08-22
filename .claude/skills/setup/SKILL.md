---
name: setup
description: Set up ExecuTorch development environment. Use when installing dependencies, setting up conda environments, or preparing to develop with ExecuTorch.
---

# Setup

1. Activate conda: `conda activate executorch`
   - If not found: `conda env list | grep -E "(executorch|et)"`

2. Install executorch: `./install_executorch.sh`

3. (Optional) For Huggingface integration:
   - Read commit from `.ci/docker/ci_commit_pins/optimum-executorch.txt`
   - Install: `pip install git+https://github.com/huggingface/optimum-executorch.git@<COMMIT>`
