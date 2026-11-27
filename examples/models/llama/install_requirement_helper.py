# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Removes lm_eval's top level `examples` package, which collides with ours.

lm_eval installs a top level package literally named `examples`. ExecuTorch's own examples
live at `executorch.examples`, but a bare `examples` on the path shadows nothing of ours
while breaking `import examples.models` for code that expects ours, so the colliding copy
is removed after installing lm_eval.

Run as a script, deliberately. The deletion used to happen at import, so anything that
imported this module, including a plugin scan that walks the installed package, deleted
whatever `examples` directory happened to be first on the path.
"""

import shutil
import sys
from pathlib import Path


def _lm_eval_examples_directory() -> Path | None:
    """The directory to remove, or None when there is nothing to remove.

    Returns a path only for a top level `examples` package that has no `models` submodule,
    which is what identifies lm_eval's copy rather than a directory that merely shares the
    name. ExecuTorch's own examples are never a candidate: they are reached as
    `executorch.examples`, not as a top level `examples`.
    """
    try:
        import examples
    except ImportError:
        return None

    try:
        import examples.models  # noqa: F401
    except ImportError:
        pass
    else:
        # Has a models submodule, so this is not the colliding copy.
        return None

    locations = list(getattr(examples, "__path__", []))
    if len(locations) != 1:
        # A namespace package spread over several directories. Which one to remove is
        # ambiguous, so remove none of them.
        return None

    directory = Path(locations[0]).resolve()
    if not (directory / "__init__.py").exists():
        # Only a regular package is removed. A namespace package's directory can be shared
        # with unrelated content.
        return None
    return directory


def main() -> int:
    directory = _lm_eval_examples_directory()
    if directory is None:
        return 0

    print(f"Removing lm_eval's colliding examples package at {directory}", flush=True)
    shutil.rmtree(directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())
