Thank you for your interest in contributing to ExecuTorch! We want to make
it easy to contribute to this project.

&nbsp;

## Dev Install

Set up your environment by following the instructions at
https://pytorch.org/executorch/stable/getting-started-setup.html to clone
the repo and install the necessary requirements.

&nbsp;

## Contributing workflow
We actively welcome your pull requests (PRs).

1. [Claim an issue](#claiming-issues), if present, before starting work. If an
   issue doesn't cover the work you plan to do, consider creating one to provide
   context about it, and to build consensus about the scope and solution.
1. Create your new branch from `main` in your forked repo, with a name
   describing the work you're completing; e.g., `add-feature-x`.
1. If you've added code that should be tested, add tests. Ensure all tests pass.
   See the [testing section](#testing) for more information.
1. If you've changed APIs or added a new tool or feature, [update the
   documentation](#updating-documentation).
1. If you added an experimental API or deprecated an existing API, follow the
   [API Life Cycle and Deprecation Policy](/docs/source/api-life-cycle.md).
1. Make sure your code follows the [style guides](#coding-style) and passes the
   [lint checks](#lintrunner).
1. If you haven't already, complete the [Contributor License Agreement ("CLA")](#contributor-license-agreement-cla).
1. Create a pull request in the `pytorch/executorch` Github repo using the
   [instructions below](#pull-requests).

&nbsp;

## Issues

### Creating Issues
We use GitHub issues to track public bugs and feature requests. Ensure that the
issue title is clear and descriptive, and that the description has sufficient
instructions to be able to reproduce the issue.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

### Claiming Issues
We'd love your help closing out [open
issues](https://github.com/pytorch/executorch/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen)
in the Github repo.

1. Find an issue with the
   [`actionable`](https://github.com/pytorch/executorch/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen+label%3Aactionable)
   or [`good first
   issue`](https://github.com/pytorch/executorch/issues?q=sort%3Aupdated-desc+is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
   label that is not currently assigned to anyone.
   - If you'd like to work on an issue that is assigned but hasn't been updated
     in a while, discuss a hand-off with the current assignee in the issue
     comments.
   - If you'd like to work on an issue that isn't marked `actionable`, please
     comment on the issue to ask about its status and wait for a response.
1. Set yourself as the assignee of the issue.
1. If you decide not to finish the issue, update the issue with information to
   help the next person, then remove yourself from the assignee list.
1. When creating pull requests (PRs), mention the issue number like `#1234` in
   the PR description details (the first comment in the PR conversation thread).
1. When the final PR has merged and resolves the issue, close the issue with the
   button at the bottom of the issue's page.

&nbsp;

## Coding Style

Goal: Encourage standards that make it easier to read, edit, maintain, and debug
the ExecuTorch code.

### lintrunner

We use [`lintrunner`](https://pypi.org/project/lintrunner/) to help make sure the
code follows our standards. Set it up with:

```
pip install lintrunner==0.11.0
pip install lintrunner-adapters==0.11.0
lintrunner init
```

Then run `lintrunner` from the root of the repo to see its suggestions, or run
`lintrunner -a` to automatically apply the suggestions.

### Python Style

ExecuTorch Python code follows the style used by the PyTorch core project.

### C++ Style

ExecuTorch code uses the [Google C++
Style](https://google.github.io/styleguide/cppguide.html), with modifications.

Rationale: Google style is close to the C++ style used by PyTorch core, although
PyTorch core does not explicitly document its C++ style. Google style is well
documented, and has exceptional tooling support.

**Modifications** to the Google C++ style, to make it closer to the code in
PyTorch core:
- Function and method names should use `lower_snake_case()`. This follows the
  convention that PyTorch core inherited from its namesake Python, and is the
  biggest modification to the Google C++ style.
- File names should use `lower_snake_case.cpp` (not `.cc`, and not
  `PascalCase.cpp`). This follows the most common pattern in PyTorch core.
- Headers should use `#pragma once` instead of manual include guards. This
  follows the most common pattern in PyTorch core.
- All includes should use `<angle brackets>`, not `"double quotes"`. This
  ensures that headers are included using the compiler's include path, and not
  relative to the local file.
- Documentation comments should follow Doxygen syntax, either `//** ... */`
  (multi-line) or `/// ...` (single line), with `@`-style parameters like
  `@param`, `@retval`. Public APIs must be documented in the `.h` files that
  declare them.
- TODOs should prefer to reference a task or issue number like `TODO(#123):
  <description>`, rather than a username. A task can manage much-more-nuanced
  information, and can change ownership as people leave and join the project.

See the rest of this file for other portability- and efficiency-related
modifications to the Google C++ style guide.

### C++ Portability Guidelines

See also [Portable C++ Programming](/docs/source/portable-cpp-programming.md)
for detailed advice.

#### C++ language version

**C++17.**

Rationale: This is a compromise between being compatible with older, proprietary
toolchains, and having access to relatively modern C++ features.

#### C/C++ standard library usage

**Restricted usage of the C++ standard library.**

Rationale: ExecuTorch is intended to be portable to bare-metal systems that lack
certain features, like dynamic memory, threading, and locking, required by parts
of the standard library. It is also intended to be as small as possible, and
some convenient stdlib features may grow the binary size unacceptably.

Generally, do not instantiate types that allocate memory under the hood, like
`std::vector` or `std::string`. Do not call `new`, `malloc()` or `mmap()`; do
not use iostreams; do not operate on files.

However, it is convenient and portable (and sometimes necessary) to use static
standard library concepts like `std::move`, or metaprogramming helpers like
`std::is_floating_point<>`.  Pure code like `<cmath>` and `<cstring>` is fine,
as long as you stay away from functions that allocate memory (like `strdup()`).

It is also allowed (and sometimes necessary) to use "placement `new`", but be
careful to also manually destroy objects initialized in this way.

#### C++ language features

**Exceptions: Do not use.**
- Rationale: Exceptions are not widely supported on some classes of
  microcontrollers and DSPs, and they can significantly increase binary size.

**Threads, thread_local, locking: Do not use, except in optional libraries that
must work with threading**
- Rationale: The core runtime must work on systems that do not have threading
  support.

**RTTI, dynamic_cast, and `<typeid>`: Do not use.**
- Rationale: RTTI adds extra data to every virtual class. ExecuTorch doesn't
  have a strong need for `dynamic_cast` and friends, so it's better to reduce
  the binary size.

**Templates and template metaprogramming: Be careful and avoid if possible.**
- Rationale: Most templating results in code generation, and is one of the most
  common sources of binary bloat. Some use of templates is fine (e.g. an
  `ArrayRef<T>`, or code that handles multiple `ScalarType` types), but for the
  most part avoid them if possible.

&nbsp;

## Testing

### Writing Tests
To help keep code quality high, ExecuTorch uses a combination of unit tests and
end-to-end (e2e) tests. If you add a new feature or fix a bug, please add tests
to ensure that the feature/fix works properly and continues to work properly.

Most directories in the repo already contain test files. In many cases, you can
add a test to an existing file, and the existing CI jobs will run it will run
automatically. If you do this, please take a look at the CI job logs to ensure
that it did actually run.

If it's not clear how to add a test for your PR, take a look at the blame for
the code you're modifying and find an author who has more context. Ask them
for their help in the PR comments.

TODO: Explain how to run tests locally without needing to push and wait for CI.

### Continuous Integration
See https://hud.pytorch.org/hud/pytorch/executorch/main for the current state of
the CI (continuous integration) jobs. If `main` is broken, consider rebasing
your PR onto the `viable/strict` branch, which points to the most recent
all-green commit.

&nbsp;

## Updating Documentation

### APIs
ExecuTorch documents its APIs using inline code comments: doc strings for
Python, and Doxygen comments for C++. When modifying or adding an API, be sure
to modify or add documentation to the interfaces that you change. If the API
doesn't have inline documentation yet, please help improve the code by adding
documentation and describing the rest of the piece you modified.

Also search for references to the API you modified under `docs/source` to see if
any docs need to be modified to reflect your changes; these are the files that
are published on https://pytorch.org/executorch. If you are adding a new API,
look for places in the docs that would benefit from talking about that API, or
even create a new document for it. A job on the PR will give you a link to a
website preview based on your changes.

&nbsp;

## Pull Requests
This repo uses Github pull requests (PRs) to stage and review code before
merging it into the `main` branch. See the [Github
docs](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork)
for basics.

1. Push your branch to your fork of `pytorch/executorch`. Most people do not
  have permission to push a branch directoy to the upstream repo.
1. Create your PR
   - Use the `main` branch as the base.
   - Give the PR a clear and descriptive title. It will become the title of the
     merged commit, so it needs to be useful in the output of `git log`.
     - Bad title: "Fix a bug"
     - Good title: "Add XYZ method to ABC"
   - Give the PR a clear and thorough description. Don't just describe what the PR
     does: the diff will do that. Explain *why* you are making this change, in a
     way that will make sense to someone years from now.
   - Add the line `Test Plan:` (with that spelling, capitalization, and trailing
     colon character), followed by lines containing repeatable instructions for
     testing the PR.
     - If you added tests, this can be as simple as the command you used to run the
       tests.
     - If you tested the PR manually, include the steps and the outputs. Help a
       future editor understand how to test the code that you're modifying
       today.
   - See https://github.com/pytorch/executorch/pull/3612 for an example PR that
     follows this advice.
1. Before asking for a review, ensure that all [CI (continuous integration)
   jobs](#continuous-integration) on your pull request succeed.
   - If the jobs on your PR are broken but you're not sure why, add a comment
     and proceed to finding a reviewer.
   - Not all users can trigger the CI jobs. If the jobs don't run on your PR,
     proceed to finding a reviewer.
1. Find reviewers
   - If you have been working with a member of the ExecuTorch repo, add them
     as a reviewer (*not* an "assignee").
   - If not, look at the blame for the files that the PR modifies, and try
     picking one or two ExecuTorch repo members as reviewers (*not*
     "assignees").
   - If you are unsure, leave a comment on the PR and keep it unassigned with no
     reviewers. A member of the ExecuTorch repo will find someone to review it.
1. Address and discuss comments left by reviewers
   - If the reviewers have requests or questions, follow up with them.
   - The goal of the reviewer is to ensure that the code in the `main` branch of
     the repo is consistent, maintainable, and of high quality.
1. Once approved, your reviewer will import the PR into Meta's internal system
   and merge it from there.
   - If the PR is approved and not merged within a few business days, please
     comment on the PR to ask about its status.
   - Note that if the `main` [CI](#continuous-integration) jobs are broken, we
     will only merge PRs that fix the broken jobs until all critical jobs are
     fixed.

&nbsp;

## For Backend Delegate Authors

- Use [this](/docs/source/backend-delegates-integration.md) guide when
  integrating your delegate with ExecuTorch.
- Refer to [this](/docs/source/backend-delegates-dependencies.md) set of
  guidelines when including a third-party depenency for your delegate.

&nbsp;

## License
By contributing to ExecuTorch, you agree that your contributions will be
licensed under the LICENSE file in the root directory of this source tree.

&nbsp;

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

&nbsp;
