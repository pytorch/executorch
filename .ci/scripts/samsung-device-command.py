#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import signal
import subprocess
import sys


def main():
    active = False
    pending_signal = None

    def interrupted(signum, _frame):
        nonlocal pending_signal
        pending_signal = signum
        if active:
            raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    process = subprocess.Popen(
        sys.argv[2:],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    completed = False
    try:
        active = True
        if pending_signal is not None:
            raise SystemExit(128 + pending_signal)
        try:
            output, _ = process.communicate(
                timeout=None if sys.argv[1] == "none" else float(sys.argv[1])
            )
        except subprocess.TimeoutExpired as error:
            output = error.output or b""
            returncode = 124
        else:
            completed = True
            returncode = process.returncode
            if returncode < 0:
                returncode = 128 - returncode
        sys.stdout.buffer.write(output)
        return returncode
    finally:
        if not completed:
            # The deadline includes pipe lifetime, even if the CLI already exited.
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            except ProcessLookupError:
                pass
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


if __name__ == "__main__":
    sys.exit(main())
