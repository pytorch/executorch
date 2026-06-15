# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest


class VersionTest(unittest.TestCase):
    def test_version_attributes_exposed(self) -> None:
        # ``import executorch`` must succeed and expose the conventional
        # ``__version__``/``git_version`` attributes (resolved from the generated
        # version.py, or a graceful fallback in an unbuilt source tree).
        import executorch

        self.assertIsInstance(executorch.__version__, str)
        self.assertTrue(executorch.__version__)
        self.assertTrue(hasattr(executorch, "git_version"))
