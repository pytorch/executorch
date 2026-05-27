import hashlib
import sys

import torch

collect_ignore_glob: list[str] = []

# Skip Apple tests on Windows. Note that some Core ML tests can run on Linux, as the AOT flow
# is available. Tests will manage this internally. However, the coremltools import is not available
# on Windows and causes collection to fail. The easiest way to manage this seems to be to just
# skip collection for this subdirectory on unsupported platforms.
if sys.platform == "win32":
    collect_ignore_glob += [
        "backends/apple/**",
    ]


def pytest_runtest_setup(item):
    # Set a stable seed for each test based on a hash of the test name.
    seed = int(hashlib.sha256(item.nodeid.encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
