# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Write a CUDA tool's output file in one atomic step.

The tools in this package rewrite model files that other steps then
load, so a half-written output must never be visible, a failed write
must not leave its temp file behind, and the result must stay
readable by whoever could read the input (a bare NamedTemporaryFile
would leave it owner-only at 0600).
"""

import os
import stat
import tempfile
from pathlib import Path
from typing import BinaryIO, Callable, Optional


def write_atomically(
    output: Path,
    write: Callable[[BinaryIO], None],
    mode_from: Path,
) -> None:
    """Write `output` via a temp file in its directory, then replace it.

    `write` receives the open temp file. The output takes
    `mode_from`'s permission bits (required, so no caller can fall
    back to the temp file's owner-only mode by omission).
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            write(temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_path, stat.S_IMODE(mode_from.stat().st_mode))
        os.replace(temporary_path, output)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
