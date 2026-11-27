# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Host memory measurements for tests (Linux: reads /proc/self/statm)."""

import os
import sys
import threading

IS_LINUX = sys.platform.startswith("linux")


def resident_bytes() -> int:
    """This process's resident set size."""
    with open("/proc/self/statm") as statm:
        return int(statm.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


class PeakResidentBytes:
    """Samples this process's resident set while the block runs."""

    def __enter__(self):
        self.base = self.peak = resident_bytes()
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._done.set()
        self._thread.join()
        self.peak = max(self.peak, resident_bytes())

    @property
    def growth(self) -> int:
        return self.peak - self.base

    def _sample(self) -> None:
        while not self._done.wait(0.01):
            self.peak = max(self.peak, resident_bytes())
