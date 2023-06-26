#!/usr/bin/env fbpython
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import unittest

from executorch.sdk.visualizer.utils import make_markdown_table

INPUT_TABLE_IN_2D_LIST = [
    ["op_name", "occurances", "avg_time", "min_time", "max_time", "p10", "p90"],
    ["aten::add", "4", "2000", "100", "3000", "200", "2500"],
    ["aten::mul", "12", "200", "10", "300", "20", "250"],
]

EXPECTED_TABLE_IN_MARKDOWN = "\
| op_name | occurances | avg_time | min_time | max_time | p10 | p90 |\n\
| ------- | ------- | ------- | ------- | ------- | ------- | ------- |\n\
| aten::add | 4 | 2000 | 100 | 3000 | 200 | 2500 |\n\
| aten::mul | 12 | 200 | 10 | 300 | 20 | 250 |"


class UtilsTest(unittest.TestCase):
    def test_make_markdown_table(self) -> None:
        self.assertEqual(
            EXPECTED_TABLE_IN_MARKDOWN, make_markdown_table(INPUT_TABLE_IN_2D_LIST)
        )
