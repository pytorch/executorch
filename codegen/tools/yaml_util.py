# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.
import yaml


class BlankLineDumper(yaml.SafeDumper):
    def write_line_break(self, data=None):
        super().write_line_break(data)
        # insert a new line between entries.
        if len(self.indents) == 1:
            super().write_line_break()
