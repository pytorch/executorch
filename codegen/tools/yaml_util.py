# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import yaml


class BlankLineDumper(yaml.SafeDumper):
    def write_line_break(self, data=None):
        super().write_line_break(data)
        # insert a new line between entries.
        if len(self.indents) == 1:
            super().write_line_break()
