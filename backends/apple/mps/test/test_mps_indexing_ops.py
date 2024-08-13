#
#  Copyright (c) 2024 Apple Inc. All rights reserved.
#  Provided subject to the LICENSE file in the top level directory.
#

import inspect

import torch
from executorch.backends.apple.mps.test.test_mps_utils import TestMPS


class TestMPSIndexingOps(TestMPS):
    def test_mps_indexing_get_1(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[[0, 1, 2], [0, 1, 0]]

        module = IndexGet()
        model_inputs = (torch.tensor([[1, 2], [3, 4], [5, 6]]),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_indexing_get_2(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[:, [0, 1, 0]]

        module = IndexGet()
        model_inputs = (torch.tensor([[1, 2], [3, 4], [5, 6]]),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_indexing_get_3(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[:, [0, 1, 0], [0, 1, 0]]

        module = IndexGet()
        model_inputs = (torch.tensor([[[1, 2], [3, 4], [5, 6]]]),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_indexing_get_4(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[:, [0, 1, 0], [0, 1, 0]]

        module = IndexGet()
        model_inputs = (
            torch.tensor([[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10], [11, 12]]]),
        )

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_indexing_get_5(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[:, [0, 4, 2]]

        module = IndexGet()
        model_inputs = (torch.randn(5, 7, 3),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_indexing_get_6(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[:, [[0, 1], [4, 3]]]

        module = IndexGet()
        model_inputs = (torch.randn(5, 7, 3),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_indexing_get_7(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[[0, 4, 2]]

        module = IndexGet()
        model_inputs = (torch.randn(5, 7, 3),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_indexing_get_8(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[[0, 2, 1], :, 0]

        module = IndexGet()
        model_inputs = (torch.ones(3, 2, 4),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_indices2d(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, rows, columns):
                return x[rows, columns]

        module = IndexGet()
        x = torch.arange(0, 12).resize(4, 3)
        rows = torch.tensor([[0, 0], [3, 3]])
        columns = torch.tensor([[0, 2], [0, 2]])
        model_inputs = (
            x,
            rows,
            columns,
        )

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_slicing_using_advanced_index_for_column_0(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                return x[1:4]

        module = IndexGet()
        model_inputs = (torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_slicing_using_advanced_index_for_column_1(self):
        class IndexGet(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                # using advanced index for column
                return x[1:4, [1, 2]]

        module = IndexGet()
        model_inputs = (torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]),)

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    # def test_boolean_array_indexing(self):
    #     class IndexGet(torch.nn.Module):
    #         def __init__(self):
    #             super().__init__()

    #         def forward(self, x):
    #             return x[x > 5]

    #     module = IndexGet()
    #     model_inputs = (torch.tensor([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]),)

    #     self.lower_and_test_with_partitioner(
    #         module, model_inputs, func_name=inspect.stack()[0].function[5:]
    #     )

    def test_mps_indexing_put_1(self):
        class IndexPut(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y, z):
                x[:, :, y] = z
                return x

        module = IndexPut()
        input = torch.ones(1, 8, 128, 8)
        indices = torch.tensor([1])
        values = torch.randn(8, 1, 8)
        model_inputs = (
            input,
            indices,
            values,
        )

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_indexing_slice_scatter_1(self):
        class IndexSliceScatter(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x.slice_scatter(y, start=6)

        module = IndexSliceScatter()
        input = torch.zeros(8, 8)
        src = torch.ones(2, 8)
        model_inputs = (
            input,
            src,
        )

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )

    def test_mps_indexing_slice_scatter_2(self):
        class IndexSliceScatter(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, x, y):
                return x.slice_scatter(y, dim=1, start=2, end=6, step=2)

        module = IndexSliceScatter()
        input = torch.zeros(8, 8)
        src = torch.ones(8, 2)
        model_inputs = (
            input,
            src,
        )

        self.lower_and_test_with_partitioner(
            module, model_inputs, func_name=inspect.stack()[0].function[5:]
        )
