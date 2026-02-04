# Repo and framework name

Refer to the repo/framework/runtime "executorch" (in lower cases) or "ExecuTorch" (in 
camel cases), not "ExecutorTorch". With limited code or comment length, maybe refer
to the framework "ET" but consider it as very unofficial and not recommended.

# Install

## Python

If the user is mostly importing `executorch` module and experimenting with Ahead-Of-Time
export flow, installation means installing `executorch` python package.

Python virtual environment or conda environment is highly recommended for installing 
executorch from source. Double check if the user wants to enable virtual enablement before
building from source.

First time install: run `install_executorch.sh` (or `install_executorch.bat` for Windows).

This script handles dependencies properly (since `executorch` depends on nightly versions
of `torch`, those packages won't be available in pip so need special index url).

Subsequent install: run `pip install . -v --no-build-isolation` inside `executorch`
directory.

Editable mode is avilable (either through `install_executorch.sh` script or `pip install . -e`.

Refer to more details in this [doc](docs/source/using-executorch-building-from-source.md).

## C++
If the user is building basic executorch C++ libraries, refer to root level [CMakeLists.txt](CMakeLists.txt).

If working with LLM/ASR runners, prefer to use [Makefile](Makefile) and cmake [presets](CMakePresets.json).

Again refer to this [doc](https://docs.pytorch.org/executorch/stable/using-executorch-building-from-source.html#building-the-c-runtime)

for more details.

# Commit messages

Don't commit unless the user explicitly asks you to.

When writing a commit message, don't make a bullet list of the individual
changes. Instead, if the PR is large, explain the order to review changes
(e.g., the logical progression), or if it's short just omit the bullet list
entirely.

Disclose that the PR was authored with Claude.

# Coding Style Guidelines

Follow these rules for all code changes in this repository:

- Minimize comments; be concise; code should be self-explanatory and self-documenting.
- Comments should be useful, for example, comments that remind the reader about
  some global context that is non-obvious and can't be inferred locally.
- Don't make trivial (1-2 LOC) helper functions that are only used once unless
  it significantly improves code readability.
- Prefer clear abstractions. State management should be explicit.
  For example, if managing state in a Python class: there should be a clear
  class definition that has all of the members: don't dynamically `setattr`
  a field on an object and then dynamically `getattr` the field on the object.
- Match existing code style and architectural patterns.
- Assume the reader has familiarity with ExecuTorch and PyTorch. They may not be the expert
  on the code that is being read, but they should have some experience in the
  area.

If uncertain, choose the simpler, more concise implementation.
