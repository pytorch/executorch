import hashlib
import platform
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

# Every test file under backends/apple/coreml that defines tests imports coremltools at module
# scope, so collection fails wherever the wheel does not declare it. Windows is already covered
# above; what is left is any Linux that is not x86_64, since coremltools publishes no build for
# those, and Python 3.14, for which it publishes no build on any platform. Keep this condition in
# sync with the coremltools marker in setup.py; .ci/scripts/tests/test_coreml_markers.py checks
# that the two agree.
_coremltools_is_declared = (
    sys.platform == "darwin"
    or (sys.platform.startswith("linux") and platform.machine() == "x86_64")
) and sys.version_info < (3, 14)

if not _coremltools_is_declared:
    collect_ignore_glob += [
        "backends/apple/coreml/**",
    ]


def pytest_runtest_setup(item):
    # Set a stable seed for each test based on a hash of the test name.
    seed = int(hashlib.sha256(item.nodeid.encode()).hexdigest(), 16) % (2**32)
    torch.manual_seed(seed)
