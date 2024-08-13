#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import sys

from typing import Optional, Sequence


# Run buck2 from the same directory (and thus repo) as this script.
BUCK_CWD: str = os.path.dirname(os.path.realpath(__file__))


class Buck2Runner:
    def __init__(self, tool_path: str) -> None:
        self._path = tool_path

    def run(self, args: Sequence[str]) -> list[str]:
        """Runs buck2 with the given args and returns its stdout as a sequence of lines."""
        try:
            cp: subprocess.CompletedProcess = subprocess.run(
                [self._path] + args, capture_output=True, cwd=BUCK_CWD, check=True
            )
            return [line.strip().decode("utf-8") for line in cp.stdout.splitlines()]
        except subprocess.CalledProcessError as ex:
            raise RuntimeError(ex.stderr.decode("utf-8")) from ex


def get_buck2_version(path: str) -> Optional[str]:
    try:
        runner = Buck2Runner(path)
        output = runner.run(["--version"])

        # Example output:
        # buck2 38f7c508bf1b87bcdc816bf56d1b9f2d2411c6be <build-id>
        #
        # We want the second value.

        return output[0].split()[1]

    except Exception as e:
        print(f"Failed to retrieve buck2 version: {e}.", file=sys.stderr)
        return None
