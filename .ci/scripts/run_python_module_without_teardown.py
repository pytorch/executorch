# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run a Python module or script while bypassing teardown on success.

The pinned PyTorch nightly can segfault while destroying MPS state after a
successful Metal export or test run. Exceptions and nonzero ``SystemExit``
codes still fail; only the interpreter's native shutdown sequence is skipped.
"""

import faulthandler
import os
import runpy
import sys
import traceback


def _exit_immediately(code: int) -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} MODULE_OR_SCRIPT [ARGS ...]", file=sys.stderr)
        _exit_immediately(2)

    target = sys.argv.pop(1)
    sys.argv[0] = target
    # Running this helper as a script puts .ci/scripts at sys.path[0], while
    # ``python -m`` puts the current working directory there. Match ``-m`` so
    # repository modules such as ``backends`` remain importable.
    sys.path[0] = os.getcwd()
    faulthandler.enable(all_threads=True)
    try:
        if os.path.isfile(target):
            runpy.run_path(target, run_name="__main__")
        else:
            runpy.run_module(target, run_name="__main__")
    except SystemExit as error:
        if error.code is None:
            code = 0
        elif isinstance(error.code, int):
            code = error.code
        else:
            print(error.code, file=sys.stderr)
            code = 1
        _exit_immediately(code)
    except Exception:
        traceback.print_exc()
        _exit_immediately(1)
    _exit_immediately(0)


if __name__ == "__main__":
    main()
