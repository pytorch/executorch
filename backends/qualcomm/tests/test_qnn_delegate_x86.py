# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Buck-native entry point that runs the QNN operator tests on the x86_64 host
simulator (no connected device, no CMake build tree).

The full ``test_qnn_delegate.py`` suite is normally driven by its argparse
``__main__`` block (``setup_environment``). That block never executes under a
Buck ``python_test`` runner -- Buck imports the module and runs the discovered
``TestCase`` subclasses directly -- so the flags that select host simulation
(``--enable_x86_64``, ``--soc_model``, ``--backend``) are never parsed. This
module re-exports the operator test classes and sets the equivalent ``TestQNN``
class attributes at import time so the same tests run unmodified under
``buck test``.

The QNN x86 SDK libraries and the host ``qnn_executor_runner`` binary are
provided by the Buck target's ``env`` (``QNN_SDK_ROOT``, ``LD_LIBRARY_PATH``,
``QNN_EXECUTOR_RUNNER``); see the ``test_qnn_delegate_x86`` target in BUCK.
"""

import os

from executorch.backends.qualcomm.tests import test_qnn_delegate as _ops
from executorch.backends.qualcomm.tests.utils import TestQNN

# Compile ahead-of-time and execute through the x86 simulator (qnn_executor_runner)
# instead of pushing to a device over adb.
TestQNN.enable_x86_64 = True
TestQNN.backend = os.environ.get("QNN_BACKEND", "htp")
# Only selects the HTP architecture baked into the offline-compiled context
# binary; the x86 simulator runs the same graph regardless of the physical SoC.
TestQNN.soc_model = os.environ.get("QNN_SOC_MODEL", "SM8650")


# Subclass (rather than re-import) so the test runner discovers these classes as
# defined in this module. The base classes stay behind the `_ops` module handle
# so they are not collected (and double-run) from this module's namespace.
class TestQNNFloatingPointOperator(_ops.TestQNNFloatingPointOperator):
    pass


class TestQNNQuantizedOperator(_ops.TestQNNQuantizedOperator):
    pass
