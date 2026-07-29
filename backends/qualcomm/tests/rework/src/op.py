# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import itertools
import random
from functools import partial, reduce
from operator import mul

import torch

from executorch.backends.qualcomm.tests.rework.conftest import (
    export_and_verify,
    temp_attribute,
)


def unpack_fixtures(func):
    def wrapper(request, kwargs):
        params = inspect.signature(func).parameters
        extra_fixtures = set(params.keys()) - set(kwargs.keys())
        new_kwargs = {key: request.getfixturevalue(key) for key in extra_fixtures}
        # hack qnn_config to get unique test folder
        with temp_attribute(
            new_kwargs["qnn_config"], "device_workspace", __name__.replace(".", "_")
        ):
            return func(**new_kwargs, **kwargs)

    return wrapper


class Abs(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.abs(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 3, 4),)
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class ACos(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.acos(x), torch.acos(y)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (
            torch.randn(1, 2, 3, 4).clamp(-1, 1),
            torch.randn(1, 2, 3, 4).clamp(-1, 1),
        )
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class AdaptiveAvgPool(torch.nn.Module):
    def __init__(self, n_dim, output_size):
        super().__init__()
        self.adaptive_avg_pool = getattr(torch.nn, f"AdaptiveAvgPool{n_dim}d")(
            output_size
        )

    def forward(self, x):
        return self.adaptive_avg_pool(x)

    @staticmethod
    @unpack_fixtures
    def test_1d_unsupported_io_shape(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 3),)
        with expected as metrics:
            export_and_verify(
                module=__class__(1, 2),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )

    @staticmethod
    @unpack_fixtures
    def test_1d(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 3),)
        with expected as metrics:
            export_and_verify(
                module=__class__(1, 1),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )

    @staticmethod
    @unpack_fixtures
    def test_2d_unsupported_io_shape(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 3, 4),)
        with expected as metrics:
            export_and_verify(
                module=__class__(2, (2, 3)),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )

    @staticmethod
    @unpack_fixtures
    def test_2d(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 4, 4),)
        for output_size in [2, (2, 2), (None, 2)]:
            with subtests.test(msg=f"output_size:{output_size}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(2, output_size),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )

    @staticmethod
    @unpack_fixtures
    def test_3d_unsupported_io_shape(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 3, 4, 5),)
        with expected as metrics:
            export_and_verify(
                module=__class__(3, (2, 3, 4)),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )

    @staticmethod
    @unpack_fixtures
    def test_3d(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 4, 4, 4),)
        for output_size in [2, (2, 2, 2), (None, 2, 2), (None, None, 2)]:
            with subtests.test(msg=f"output_size:{output_size}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(3, output_size),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class AdaptiveMaxPool(torch.nn.Module):
    def __init__(self, n_dim, output_size, return_indices):
        super().__init__()
        self.adaptive_max_pool = getattr(torch.nn, f"AdaptiveMaxPool{n_dim}d")(
            output_size=output_size,
            return_indices=return_indices,
        )

    def forward(self, x):
        return self.adaptive_max_pool(x)

    @staticmethod
    @unpack_fixtures
    def test_2d(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 4, 4),)
        for output_size in [2, (2, 2), (None, 2)]:
            with subtests.test(msg=f"output_size:{output_size}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(
                            n_dim=2, output_size=output_size, return_indices=False
                        ),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )

    @staticmethod
    @unpack_fixtures
    def test_2d_with_indices(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 4, 4),)
        with expected as metrics:
            export_and_verify(
                module=__class__(n_dim=2, output_size=(2, 2), return_indices=True),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class Add(torch.nn.Module):
    def __init__(self, alpha, constant=None):
        super().__init__()
        self.alpha = alpha
        self.add = torch.add if constant is None else partial(torch.add, constant)

    def forward(self, *args):
        return self.add(*args, alpha=self.alpha)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        for alpha in [1, 2]:
            with subtests.test(msg="(tensor, tensor)"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(alpha=alpha),
                        inputs=(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )
            with subtests.test(msg="(constant, tensor)"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(constant=2, alpha=alpha),
                        inputs=(torch.randn(1, 2, 3, 4),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class AddMM(torch.nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, bias, input, mat2):
        return torch.addmm(bias, input, mat2, alpha=self.alpha, beta=self.beta)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        for alpha, beta in [(1, 2), (2, 1), (2, 3)]:
            with subtests.test(msg=f"alpha={alpha}, beta={beta}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(alpha=alpha, beta=beta),
                        inputs=(
                            torch.randn(8),
                            torch.randn(4, 3),
                            torch.randn(3, 8),
                        ),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Alias(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        alias_x = torch.ops.aten.alias.default(x)
        return self.relu(alias_x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=(torch.randn(1, 10),),
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class AMax(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.amax(x, dim=self.dim, keepdim=self.keepdim)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        dims = [[0, 1], 1]
        keepdims = [False, True]

        for dim, keepdim in itertools.product(dims, keepdims):
            with subtests.test(msg=f"dim:{dim}, keepdim:{keepdim}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim=dim, keepdim=keepdim),
                        inputs=(torch.randn(1, 2, 3, 4),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )
        with subtests.test(msg="all_reduce"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(dim=None, keepdim=False),
                    inputs=(torch.randn(1, 2, 3, 4),),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )


class AMin(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.amin(x, dim=self.dim, keepdim=self.keepdim)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        dims = [[0, 1], 1]
        keepdims = [False, True]

        for dim, keepdim in itertools.product(dims, keepdims):
            with subtests.test(msg=f"dim:{dim}, keepdim:{keepdim}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim=dim, keepdim=keepdim),
                        inputs=(torch.randn(1, 2, 3, 4),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )
        with subtests.test(msg="all_reduce"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(dim=None, keepdim=False),
                    inputs=(torch.randn(1, 2, 3, 4),),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )


class Any(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.any(x, dim=self.dim, keepdim=self.keepdim)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        dims = [None, [0, 1], 1]
        keepdims = [False, True]

        for dim, keepdim in itertools.product(dims, keepdims):
            with subtests.test(msg=f"dim:{dim}, keepdim:{keepdim}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim=dim, keepdim=keepdim),
                        inputs=(torch.randn(1, 2, 3, 4) > 0,),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Arange(torch.nn.Module):
    def __init__(self, start, end, step, dtype):
        super().__init__()
        self.start = start
        self.end = end
        self.step = step
        self.dtype = dtype

    def forward(self, x):
        return (
            torch.arange(
                start=self.start, end=self.end, step=self.step, dtype=self.dtype
            )
            + x
        )

    @staticmethod
    @unpack_fixtures
    def test_dtype_int(qnn_config, quantizer, compile_spec, expected):
        with expected as metrics:
            export_and_verify(
                module=__class__(start=0, end=4, step=1, dtype=torch.int32),
                inputs=(torch.randint(0, 10, (1, 2, 3, 4)),),
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )

    @staticmethod
    @unpack_fixtures
    def test_dtype_float(qnn_config, quantizer, compile_spec, expected):
        with expected as metrics:
            export_and_verify(
                module=__class__(start=0, end=4, step=1, dtype=torch.float),
                inputs=(torch.randn(1, 2, 3, 4),),
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class ArgMax(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.argmax(x, dim=self.dim, keepdim=self.keepdim)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        dims = [1, -1]
        keepdims = [False, True]

        for dim, keepdim in itertools.product(dims, keepdims):
            with subtests.test(msg=f"dim:{dim}, keepdim:{keepdim}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim=dim, keepdim=keepdim),
                        inputs=(torch.randn(1, 2, 3, 4),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )
        with subtests.test(msg="all_reduce"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(dim=None, keepdim=False),
                    inputs=(torch.randn(1, 2, 3, 4),),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )


class ArgMin(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.argmin(x, dim=self.dim, keepdim=self.keepdim)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        dims = [1, -1]
        keepdims = [False, True]

        for dim, keepdim in itertools.product(dims, keepdims):
            with subtests.test(msg=f"dim:{dim}, keepdim:{keepdim}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim=dim, keepdim=keepdim),
                        inputs=(torch.randn(1, 2, 3, 4),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )
        with subtests.test(msg="all_reduce"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(dim=None, keepdim=False),
                    inputs=(torch.randn(1, 2, 3, 4),),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )


class ASin(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.asin(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=(torch.randn(1, 2, 3, 4).clamp(-1, 1),),
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class ATan(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.atan(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=(torch.randn(1, 2, 3, 4),),
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class ATan2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y1, x1, y2, x2):
        return torch.atan2(y1, x1), torch.atan2(y2, x2)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = [
            (
                torch.randn(3, 4),
                torch.randn(3, 4).abs(),
                torch.randn(3, 4),
                torch.randn(3, 4),
            ),
            (
                torch.randn(3, 4).abs(),
                -torch.randn(3, 4).abs(),
                -torch.randn(3, 4).abs(),
                -torch.randn(3, 4).abs(),
            ),
            (
                -torch.randn(3, 4).abs() - 1e-3,
                torch.zeros(3, 4),
                torch.randn(3, 4).abs() + 1e-3,
                torch.zeros(3, 4),
            ),
        ]
        for index, input in enumerate(inputs):
            with subtests.test(msg=index):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(),
                        inputs=input,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


# TODO: issues required to be addressed:
# 1. ceil mode will make the output shape misaligned with pytorch
#    definition, currently the test cases are designed to genereate
#    deterministic output shape
# 2. combinations of "padding / count_include_pad" are having unexpected
#    behavior, need further investigation
class AvgPool(torch.nn.Module):
    def __init__(
        self, n_dim, kernel_size, stride, padding, ceil_mode, count_include_pad
    ):
        super().__init__()
        self.avg_pool = getattr(torch.nn, f"AvgPool{n_dim}d")(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )

    def forward(self, x):
        return self.avg_pool(x)

    @staticmethod
    def _test(
        n_dim,
        subtests,
        qnn_config,
        quantizer,
        compile_spec,
        expected,
        inputs,
        kernel_sizes,
        strides,
        paddings,
        ceil_modes,
        count_include_pads,
    ):
        configs = list(
            itertools.product(
                kernel_sizes,
                strides,
                paddings,
                ceil_modes,
                count_include_pads,
            )
        )
        for kernel_size, stride, padding, ceil_mode, count_include_pad in configs:
            with subtests.test(
                msg=(
                    f"kernel:{kernel_size}, stride:{stride}, padding:{padding} ,"
                    f"ceil_mode:{ceil_mode}, count_include_pad:{count_include_pad}"
                )
            ):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(
                            n_dim=n_dim,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            ceil_mode=ceil_mode,
                            count_include_pad=count_include_pad,
                        ),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )

    @staticmethod
    @unpack_fixtures
    def test_1d(subtests, qnn_config, quantizer, compile_spec, expected):
        kernel_sizes = [2]
        strides = [1, 2]
        paddings = [0, (1,)]
        ceil_modes = [False, True]
        count_include_pads = [False, True]
        inputs = (torch.randn(1, 2, 6),)
        __class__._test(
            n_dim=1,
            subtests=subtests,
            qnn_config=qnn_config,
            quantizer=quantizer,
            compile_spec=compile_spec,
            expected=expected,
            inputs=inputs,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            ceil_modes=ceil_modes,
            count_include_pads=count_include_pads,
        )

    @staticmethod
    @unpack_fixtures
    def test_2d(subtests, qnn_config, quantizer, compile_spec, expected):
        kernel_sizes = [2]
        strides = [1, (2, 2)]
        paddings = [0, (1, 1)]
        ceil_modes = [False, True]
        count_include_pads = [False, True]
        inputs = (torch.randn(1, 2, 6, 6),)
        __class__._test(
            n_dim=2,
            subtests=subtests,
            qnn_config=qnn_config,
            quantizer=quantizer,
            compile_spec=compile_spec,
            expected=expected,
            inputs=inputs,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            ceil_modes=ceil_modes,
            count_include_pads=count_include_pads,
        )

    @staticmethod
    @unpack_fixtures
    def test_3d(subtests, qnn_config, quantizer, compile_spec, expected):
        kernel_sizes = [2]
        strides = [1, (2, 2, 2)]
        paddings = [0, (1, 1, 1)]
        ceil_modes = [False, True]
        count_include_pads = [False, True]
        inputs = (torch.randn(1, 2, 6, 6, 6),)
        __class__._test(
            n_dim=3,
            subtests=subtests,
            qnn_config=qnn_config,
            quantizer=quantizer,
            compile_spec=compile_spec,
            expected=expected,
            inputs=inputs,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            ceil_modes=ceil_modes,
            count_include_pads=count_include_pads,
        )


class BatchNorm2d(torch.nn.Module):
    def __init__(self, n_features, affine):
        super().__init__()
        self.batchnorm = torch.nn.BatchNorm2d(num_features=n_features, affine=affine)

    def forward(self, x):
        return self.batchnorm(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 3, 4),)
        for affine in [False, True]:
            with subtests.test(msg=f"affine:{affine}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(n_features=2, affine=affine),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class BitwiseOp(torch.nn.Module):
    def __init__(self, op):
        super().__init__()
        self.bitwise_op = getattr(torch, f"bitwise_{op}")

    def forward(self, x, y):
        return self.bitwise_op(x, y)

    @staticmethod
    def _test_numeric(qnn_config, quantizer, compile_spec, expected, op):
        inputs = (
            torch.randint(0, 10, (1, 2, 3, 4)),
            torch.randint(0, 10, (1, 2, 3, 4)),
        )
        with expected as metrics:
            export_and_verify(
                module=__class__(op),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )

    @staticmethod
    def _test_bool(qnn_config, quantizer, compile_spec, expected, op):
        inputs = (
            torch.empty(1, 2, 3, 4, dtype=torch.bool).random_(),
            torch.empty(1, 2, 3, 4, dtype=torch.bool).random_(),
        )
        with expected as metrics:
            export_and_verify(
                module=__class__(op),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )

    @staticmethod
    @unpack_fixtures
    def test_and_numeric(qnn_config, quantizer, compile_spec, expected):
        __class__._test_numeric(qnn_config, quantizer, compile_spec, expected, op="and")

    @staticmethod
    @unpack_fixtures
    def test_and_bool(qnn_config, quantizer, compile_spec, expected):
        __class__._test_bool(qnn_config, quantizer, compile_spec, expected, op="and")

    @staticmethod
    @unpack_fixtures
    def test_or_numeric(qnn_config, quantizer, compile_spec, expected):
        __class__._test_numeric(qnn_config, quantizer, compile_spec, expected, op="or")

    @staticmethod
    @unpack_fixtures
    def test_or_bool(qnn_config, quantizer, compile_spec, expected):
        __class__._test_bool(qnn_config, quantizer, compile_spec, expected, op="or")

    @staticmethod
    @unpack_fixtures
    def test_xor_numeric(qnn_config, quantizer, compile_spec, expected):
        __class__._test_numeric(qnn_config, quantizer, compile_spec, expected, op="xor")

    @staticmethod
    @unpack_fixtures
    def test_xor_bool(qnn_config, quantizer, compile_spec, expected):
        __class__._test_bool(qnn_config, quantizer, compile_spec, expected, op="xor")


class Bmm(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.bmm(x, y)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 3), torch.randn(1, 3, 4))
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class Cast(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        return x.to(self.dtype)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = [
            (torch.randn(1, 2, 3, 4),),
            (torch.randint(-10, 10, (1, 2, 3, 4)),),
        ]
        dtypes = [torch.long, torch.float]
        for input, dtype in zip(inputs, dtypes):
            with subtests.test(msg=f"to dtype:{dtype}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dtype=dtype),
                        inputs=input,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Cat(torch.nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, x, y):
        return torch.cat((x, y), axis=self.axis)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4))
        axes = [-1, 1]
        for axis in axes:
            with subtests.test(msg=f"axis:{axis}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(axis=axis),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class CDist(torch.nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = 2

    def forward(self, x, y):
        return torch.cdist(x, y, p=self.p)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = [
            (torch.randn(1, 2, 3), torch.randn(1, 4, 3)),
        ]
        ps = [2]
        for input, p in itertools.product(inputs, ps):
            with subtests.test(msg=f"inputs_shape:{[x.shape for x in input]}, p:{p}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(p=p),
                        inputs=input,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Ceil(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.ceil(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 4, 3, 3),)
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class ChannelShuffle(torch.nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.channel_shuffle = torch.nn.ChannelShuffle(groups)

    def forward(self, x):
        return self.channel_shuffle(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 4, 3, 3),)
        with expected as metrics:
            export_and_verify(
                module=__class__(groups=2),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class Chunk(torch.nn.Module):
    def __init__(self, chunks, dim):
        super().__init__()
        self.chunks = chunks
        self.dim = dim

    def forward(self, x):
        return torch.chunk(x, chunks=self.chunks, dim=self.dim)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = [
            (torch.randn(1, 2, 11, 11),),
            (torch.randn(1, 2, 12, 12),),
            (torch.randn(1, 2, 13, 13),),
        ]
        dims = [-1, 2]
        for input, dim in itertools.product(inputs, dims):
            with subtests.test(
                msg=f"inputs_shape:{[x.shape for x in input]}, dim:{dim}"
            ):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(chunks=6, dim=dim),
                        inputs=input,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Clamp(torch.nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, min=self.min, max=self.max)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 3, 4),)
        with expected as metrics:
            export_and_verify(
                module=__class__(min=-1, max=1),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class ClampMax(torch.nn.Module):
    def __init__(self, max):
        super().__init__()
        self.max = max

    def forward(self, x):
        return torch.clamp_max(x, max=self.max)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 3, 4),)
        with expected as metrics:
            export_and_verify(
                module=__class__(max=1),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class ClampMin(torch.nn.Module):
    def __init__(self, min):
        super().__init__()
        self.min = min

    def forward(self, x):
        return torch.clamp_min(x, min=self.min)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 3, 4),)
        with expected as metrics:
            export_and_verify(
                module=__class__(min=-1),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class Clone(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.clone(x) + y

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)),
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class Conv(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        variant,
        transpose,
    ):
        super().__init__()
        module = getattr(
            torch.nn,
            (f"ConvTranspose{variant}d" if transpose else f"Conv{variant}d"),
        )
        self.conv = module(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)

    @staticmethod
    def _test(
        subtests,
        qnn_config,
        quantizer,
        compile_spec,
        expected,
        inputs,
        in_channels,
        out_channels,
        kernel_size,
        strides,
        paddings,
        dilations,
        groups,
        biases,
        variant,
        transpose,
    ):
        configs = itertools.product(strides, paddings, dilations, groups, biases)
        for stride, padding, dilation, group, bias in configs:
            with subtests.test(
                msg=(
                    f"stride:{stride}, padding:{padding}, dilation:{dilation}, "
                    f"group:{group}, bias:{bias}"
                )
            ):
                module = __class__(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                    groups=group,
                    bias=bias,
                    variant=variant,
                    transpose=transpose,
                )
                with expected as metrics:
                    export_and_verify(
                        module=module,
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )

    @staticmethod
    @unpack_fixtures
    def test_1d(subtests, qnn_config, quantizer, compile_spec, expected):
        __class__._test(
            subtests,
            qnn_config,
            quantizer,
            compile_spec,
            expected,
            inputs=(torch.randn(1, 6, 8),),
            in_channels=6,
            out_channels=8,
            kernel_size=2,
            strides=[1, 2],
            paddings=[0, 1],
            dilations=[1, 2],
            groups=[1, 2],
            biases=[False, True],
            variant=1,
            transpose=False,
        )

    @staticmethod
    @unpack_fixtures
    def test_1d_transpose(subtests, qnn_config, quantizer, compile_spec, expected):
        __class__._test(
            subtests,
            qnn_config,
            quantizer,
            compile_spec,
            expected,
            inputs=(torch.randn(1, 6, 8),),
            in_channels=6,
            out_channels=8,
            kernel_size=2,
            strides=[1, 2],
            paddings=[0, 1],
            dilations=[1, 2],
            groups=[1, 2],
            biases=[False, True],
            variant=1,
            transpose=True,
        )

    @staticmethod
    @unpack_fixtures
    def test_2d(subtests, qnn_config, quantizer, compile_spec, expected):
        __class__._test(
            subtests,
            qnn_config,
            quantizer,
            compile_spec,
            expected,
            inputs=(torch.randn(1, 6, 8, 8),),
            in_channels=6,
            out_channels=8,
            kernel_size=(2, 2),
            strides=[1, 2],
            paddings=[0, 1],
            dilations=[1, 2],
            groups=[1, 2],
            biases=[False, True],
            variant=2,
            transpose=False,
        )

    @staticmethod
    @unpack_fixtures
    def test_2d_transpose(subtests, qnn_config, quantizer, compile_spec, expected):
        __class__._test(
            subtests,
            qnn_config,
            quantizer,
            compile_spec,
            expected,
            inputs=(torch.randn(1, 6, 8, 8),),
            in_channels=6,
            out_channels=8,
            kernel_size=(2, 2),
            strides=[1, 2],
            paddings=[0, 1],
            dilations=[1, 2],
            groups=[1, 2],
            biases=[False, True],
            variant=2,
            transpose=True,
        )

    @staticmethod
    @unpack_fixtures
    def test_2d_linear_like(subtests, qnn_config, quantizer, compile_spec, expected):
        __class__._test(
            subtests,
            qnn_config,
            quantizer,
            compile_spec,
            expected,
            inputs=(torch.randn(1, 128, 8, 8),),
            in_channels=128,
            out_channels=32,
            kernel_size=(1, 1),
            strides=[1],
            paddings=[0],
            dilations=[1],
            groups=[1],
            biases=[False, True],
            variant=2,
            transpose=False,
        )

    @staticmethod
    @unpack_fixtures
    def test_3d(subtests, qnn_config, quantizer, compile_spec, expected):
        __class__._test(
            subtests,
            qnn_config,
            quantizer,
            compile_spec,
            expected,
            inputs=(torch.randn(1, 6, 8, 8, 8),),
            in_channels=6,
            out_channels=8,
            kernel_size=(2, 2, 2),
            strides=[1, 2],
            paddings=[0, 1],
            # TODO: extend this when QNN starts to support dilation
            dilations=[1],
            groups=[1, 2],
            biases=[False, True],
            variant=3,
            transpose=False,
        )

    @staticmethod
    @unpack_fixtures
    def test_3d_transpose(subtests, qnn_config, quantizer, compile_spec, expected):
        __class__._test(
            subtests,
            qnn_config,
            quantizer,
            compile_spec,
            expected,
            inputs=(torch.randn(1, 6, 8, 8, 8),),
            in_channels=6,
            out_channels=8,
            kernel_size=(2, 2, 2),
            strides=[1, 2],
            paddings=[0, 1],
            # TODO: extend this when QNN starts to support dilation
            dilations=[1],
            groups=[1, 2],
            biases=[False, True],
            variant=3,
            transpose=True,
        )


class Cos(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cos(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=(torch.randn(1, 2, 3, 4),),
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class CumSum(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.cumsum(x, dim=self.dim)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        for dim in [-1, 2]:
            with subtests.test(msg=f"dim:{dim}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim=dim),
                        inputs=(torch.randn(1, 2, 3, 4),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Div(torch.nn.Module):
    def __init__(self, constant=None):
        super().__init__()
        self.div = torch.div if constant is None else partial(torch.div, constant)

    def forward(self, *args):
        return self.div(*args)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        with subtests.test(msg="(tensor, tensor)"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(),
                    inputs=(
                        torch.randn(1, 2, 3, 4),
                        torch.randn(1, 2, 3, 4).abs() + 0.1,
                    ),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )
        with subtests.test(msg="(constant, tensor)"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(constant=2),
                    inputs=(torch.randn(1, 2, 3, 4),),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )


class DivWithRoundingMode(torch.nn.Module):
    def __init__(self, rounding_mode):
        super().__init__()
        self.rounding_mode = rounding_mode
        self.scalar = 2.0

    def forward(self, *x):
        return (
            torch.div(x[0], x[1], rounding_mode=self.rounding_mode)
            if len(x) > 1
            else torch.div(x[0], self.scalar, rounding_mode=self.rounding_mode)
        )

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = [
            (torch.randn(2, 3), torch.randn(2, 3).abs() + 1e-6),
            (torch.randn(2, 3),),
        ]
        for input, mode in itertools.product(inputs, [None, "trunc", "floor"]):
            with subtests.test(msg=f"rounding_mode={mode}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(rounding_mode=mode),
                        inputs=input,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Einsum(torch.nn.Module):
    def __init__(self, equation):
        super().__init__()
        self.equation = equation

    def forward(self, *args):
        return torch.einsum(self.equation, *args)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        equations = ["bn,anm,bm->ba", "bij,bjk->bik"]
        inputs = [
            (torch.randn(2, 5), torch.randn(3, 5, 4), torch.randn(2, 4)),
            (torch.randn(3, 2, 5), torch.randn(3, 5, 4)),
        ]
        for equation, input in zip(equations, inputs):
            with subtests.test(msg=f"equation:{equation}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(equation=equation),
                        inputs=input,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Elu(torch.nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.elu = torch.nn.ELU(alpha=alpha)

    def forward(self, x):
        return self.elu(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        for alpha in [0.5, 0.7]:
            with subtests.test(msg=f"alpha:{alpha}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(alpha=alpha),
                        inputs=(torch.randn(1, 2, 3, 4),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Embedding(torch.nn.Module):
    def __init__(self, num_embedding, embedding_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(
            num_embeddings=num_embedding,
            embedding_dim=embedding_dim,
        )

    def forward(self, x):
        return self.embedding(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        num_embedding, embedding_dim = 10, 3
        for inputs in [
            (torch.randint(0, num_embedding, (1, 2)),),
            (torch.randint(0, num_embedding, (2, 3)),),
        ]:
            with subtests.test(
                msg=f"num_embedding:{num_embedding}, input_shape:{[x.shape for x in inputs]}"
            ):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(
                            num_embedding=num_embedding,
                            embedding_dim=embedding_dim,
                        ),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Equal(torch.nn.Module):
    def __init__(self, constant=None):
        super().__init__()
        self.constant = constant

    def forward(self, *args):
        if self.constant is not None:
            return args[0] == self.constant
        return args[0] == args[1]

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        with subtests.test(msg="(tensor, tensor)"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(),
                    inputs=(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )
        with subtests.test(msg="(constant, tensor)"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(constant=0),
                    inputs=(torch.randn(1, 2, 3, 4),),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )


class Expand(torch.nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.sizes = sizes

    def forward(self, x):
        return x.expand(*self.sizes)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        for sizes in [(1, 2, 3, 4), (-1, -1, -1, 4)]:
            with subtests.test(msg=f"sizes={sizes}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(sizes=sizes),
                        inputs=(torch.randn(1, 2, 3, 1),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class ExpandAs(torch.nn.Module):
    def __init__(self, constant=None):
        super().__init__()
        self.constant = constant

    def forward(self, *args):
        if self.constant is not None:
            return self.constant.expand_as(args[0])
        return args[1].expand_as(args[0])

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        with subtests.test(msg="(tensor, tensor)"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(),
                    inputs=(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 1)),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )
        with subtests.test(msg="(constant, tensor)"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(constant=torch.ones(1, 2, 3, 1)),
                    inputs=(torch.randn(1, 2, 3, 4),),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )


class Exp(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.exp(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=(torch.randn(1, 2, 3, 4),),
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class ExpM1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.special.expm1(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=(torch.randn(1, 2, 3, 4),),
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class Fill(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.add(x, torch.fill(x, self.value))

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        with expected as metrics:
            export_and_verify(
                module=__class__(value=3.14),
                inputs=(torch.randn(1, 2, 3, 4),),
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class Flip(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return torch.flip(x, dims=self.dims)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        for dims in [(1, 2), (0, 1, -1)]:
            with subtests.test(msg=f"dims={dims}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dims=dims),
                        inputs=(torch.randn(1, 2, 3, 4),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Floor(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.floor(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=(torch.randn(1, 2, 3, 4),),
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class FloorDivide(torch.nn.Module):
    def __init__(self, constant=None):
        super().__init__()
        self.floor_divide = (
            torch.floor_divide
            if constant is None
            else partial(torch.floor_divide, constant)
        )

    def forward(self, *args):
        return self.floor_divide(*args)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        with subtests.test(msg="(tensor, tensor)"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(),
                    inputs=(
                        torch.randn(1, 2, 3, 4),
                        torch.randn(1, 2, 3, 4).abs() + 0.1,
                    ),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )
        with subtests.test(msg="(constant, tensor)"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(constant=2),
                    inputs=(torch.randn(1, 2, 3, 4),),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )


class Fold(torch.nn.Module):
    def __init__(self, output_size, kernel_size, dilation, padding, stride):
        super().__init__()
        self.fold = torch.nn.Fold(
            output_size=output_size,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            stride=stride,
        )

    def forward(self, x):
        return self.fold(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        dilations = [1]
        paddings = [0]
        strides = [1, 2, 3]
        configs = itertools.product(dilations, paddings, strides)
        num_channels, output_size = 5, (18, 36)
        for dilation, padding, stride in configs:
            with subtests.test(
                msg=(
                    f"output_size:{output_size}, kernel_size:{stride}, "
                    f"dilation:{dilation}, padding:{padding}, stride:{stride}"
                )
            ):
                kernel_size = stride
                blocks = [
                    (o_sz + 2 * padding - dilation * (kernel_size - 1) - 1) // stride
                    + 1
                    for o_sz in output_size
                ]
                module = __class__(
                    output_size=output_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    stride=stride,
                )
                inputs = (
                    torch.randn(
                        1, num_channels * (kernel_size**2), reduce(mul, blocks)
                    ),
                )
                with expected as metrics:
                    export_and_verify(
                        module=module,
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )

    @staticmethod
    @unpack_fixtures
    def test_unsupported_parameters(
        subtests, qnn_config, quantizer, compile_spec, expected
    ):
        configs = [
            # dilation, padding, stride, kernel_size
            (2, 0, (1, 1), (2, 2)),
            (1, 1, (1, 1), (2, 2)),
            (1, 0, (2, 2), (2, 3)),
            (1, 0, (2, 3), (2, 2)),
            (1, 0, (2, 3), (2, 3)),
        ]
        num_channels, output_size = 5, (18, 36)
        for dilation, padding, stride, kernel_size in configs:
            with subtests.test(
                msg=(
                    f"output_size:{(output_size)}, kernel_size:{kernel_size}, "
                    f"dilation:{dilation}, padding:{padding}, stride:{stride}"
                )
            ):
                blocks = [
                    (output_size[i] + 2 * padding - dilation * (kernel_size[i] - 1) - 1)
                    // stride[i]
                    + 1
                    for i in range(2)
                ]
                module = __class__(
                    output_size=output_size,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=padding,
                    stride=stride,
                )
                inputs = (
                    torch.randn(
                        1, num_channels * reduce(mul, kernel_size), reduce(mul, blocks)
                    ),
                )
                with expected as metrics:
                    export_and_verify(
                        module=module,
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Full(torch.nn.Module):
    def __init__(self, fill_value):
        super().__init__()
        self.fill_value = fill_value

    def forward(self, x):
        return x + torch.full(size=x.shape, fill_value=self.fill_value)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        with expected as metrics:
            export_and_verify(
                module=__class__(fill_value=1.0),
                inputs=(torch.randn(1, 2, 3, 4),),
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class FullLike(torch.nn.Module):
    def __init__(self, fill_value):
        super().__init__()
        self.fill_value = fill_value

    def forward(self, x):
        return x + torch.full_like(x, self.fill_value)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        with expected as metrics:
            export_and_verify(
                module=__class__(fill_value=1.0),
                inputs=(torch.randn(1, 2, 3, 4),),
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class Gather(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, index):
        return torch.gather(x, dim=self.dim, index=index)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        for dim in [-1, 2]:
            with subtests.test(msg=f"dim={dim}"):
                input = torch.randn(1, 2, 3, 4)
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim=dim),
                        inputs=(input, torch.randint(0, input.shape[dim], input.shape)),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Gelu(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.GELU()(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=(torch.randn(1, 2, 3, 4),),
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class Glu(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.glu = torch.nn.GLU(dim=dim)

    def forward(self, x):
        return self.glu(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        for dim in [-1, 1]:
            with subtests.test(msg=f"dim={dim}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim=dim),
                        inputs=(torch.randn(1, 2, 3, 4),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Greater(torch.nn.Module):
    def __init__(self, greater_equal, constant=None):
        super().__init__()
        self.cmp = torch.ge if greater_equal else torch.gt
        self.constant = constant

    def forward(self, *args):
        if self.constant is not None:
            return self.cmp(args[0], self.constant)
        return self.cmp(args[0], args[1])

    @staticmethod
    @unpack_fixtures
    def test_gt(subtests, qnn_config, quantizer, compile_spec, expected):
        with subtests.test(msg="(tensor, tensor)"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(greater_equal=False),
                    inputs=(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )
        with subtests.test(msg="(constant, tensor)"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(greater_equal=False, constant=1.0),
                    inputs=(torch.randn(1, 2, 3, 4),),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )

    @staticmethod
    @unpack_fixtures
    def test_ge(subtests, qnn_config, quantizer, compile_spec, expected):
        with subtests.test(msg="(tensor, tensor)"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(greater_equal=True),
                    inputs=(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )
        with subtests.test(msg="(constant, tensor)"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(greater_equal=True, constant=1.0),
                    inputs=(torch.randn(1, 2, 3, 4),),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )


class GridSample(torch.nn.Module):
    def __init__(self, mode, padding_mode, align_corners):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode
        self.align_corners = align_corners

    def forward(self, x, grid):
        grid_sample = torch.nn.functional.grid_sample(
            x, grid, self.mode, self.padding_mode, self.align_corners
        )
        return grid_sample

    @staticmethod
    def _test(subtests, qnn_config, quantizer, compile_spec, expected, inputs):
        modes = ["bilinear", "nearest"]
        padding_modes = ["zeros", "border", "reflection"]
        align_corners = [False, True]

        for mode, padding_mode, align_corner in itertools.product(
            modes, padding_modes, align_corners
        ):
            with subtests.test(
                msg=(
                    f"mode:{mode}, padding_mode:{padding_mode}, "
                    f"align_corners:{align_corner}"
                )
            ):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(
                            mode=mode,
                            padding_mode=padding_mode,
                            align_corners=align_corner,
                        ),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )

    @staticmethod
    @unpack_fixtures
    def test_4d(subtests, qnn_config, quantizer, compile_spec, expected):
        __class__._test(
            subtests=subtests,
            qnn_config=qnn_config,
            quantizer=quantizer,
            compile_spec=compile_spec,
            expected=expected,
            inputs=(torch.randn(1, 2, 3, 4), torch.randn(1, 3, 4, 2).clamp(-1, 1)),
        )

    @staticmethod
    @unpack_fixtures
    def test_5d(subtests, qnn_config, quantizer, compile_spec, expected):
        __class__._test(
            subtests=subtests,
            qnn_config=qnn_config,
            quantizer=quantizer,
            compile_spec=compile_spec,
            expected=expected,
            inputs=(
                torch.randn(1, 2, 3, 4, 5),
                torch.randn(1, 3, 4, 5, 3).clamp(-1, 1),
            ),
        )


class GroupNorm(torch.nn.Module):
    def __init__(self, num_groups, num_channels, eps, affine):
        super().__init__()
        self.norm = torch.nn.GroupNorm(
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine,
        )

    def forward(self, x):
        return self.norm(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        num_groups = [1, 3]
        num_channel = 6
        epsilons = [1e-5, 1e-2]
        affines = [False]

        for num_group, eps, affine in itertools.product(num_groups, epsilons, affines):
            with subtests.test(
                msg=(
                    f"num_groups:{num_group}, num_channels:{num_channel}, "
                    f"eps:{eps}, affine:{affine}"
                )
            ):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(
                            num_groups=num_group,
                            num_channels=num_channel,
                            eps=eps,
                            affine=affine,
                        ),
                        inputs=(torch.randn(1, num_channel, 3, 4),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class HardSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.Hardsigmoid()(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=(torch.randn(1, 2, 3, 4),),
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class HardSwish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.Hardswish()(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=(torch.randn(1, 2, 3, 4),),
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class HardTanh(torch.nn.Module):
    def __init__(self, min_val, max_val):
        super().__init__()
        self.hardtanh = torch.nn.Hardtanh(min_val=min_val, max_val=max_val)

    def forward(self, x):
        return self.hardtanh(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        for bound in [1.0, 2.0]:
            with subtests.test(msg=f"min_val:{-bound}, max_val:{bound}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(min_val=-bound, max_val=bound),
                        inputs=(torch.randn(1, 2, 3, 4),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Index(torch.nn.Module):
    def __init__(self, axis):
        super().__init__()
        self.idx0 = torch.tensor([[0, 1], [2, 3], [4, 5]], dtype=torch.int32)
        self.idx1 = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32)
        self.axis = axis
        self.dispatcher = {
            0: lambda x: x[self.idx0] + x[self.idx1],
            1: lambda x: x[:, self.idx0] + x[:, self.idx1],
            2: lambda x: x[:, :, self.idx0] + x[:, :, self.idx1],
        }

    def forward(self, x):
        return self.dispatcher[self.axis](x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn([8, 172, 64]),)
        for axis in [0, 1, 2]:
            with subtests.test(msg=f"axis:{axis}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(axis=axis),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class IndexCopy(torch.nn.Module):
    def __init__(self, copy_dim):
        super().__init__()
        self.copy_dim = copy_dim
        self.register_buffer(
            "k_cache",
            torch.zeros((1, 1024, 12, 64), dtype=torch.float32),
            persistent=True,
        )

    def forward(self, input_pos, k_val):
        k_out = self.k_cache
        k_out.index_copy_(self.copy_dim, input_pos, k_val)
        return k_out + 0

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        cases = [
            (
                1,
                (
                    torch.tensor([2], dtype=torch.int64),
                    torch.randn([1, 1, 12, 64]),
                ),
            ),
            (
                2,
                (
                    torch.tensor([2], dtype=torch.int64),
                    torch.randn([1, 1024, 1, 64]),
                ),
            ),
            (
                2,
                (
                    torch.tensor([2, 5], dtype=torch.int64),
                    torch.randn([1, 1024, 2, 64]),
                ),
            ),
        ]
        for copy_dim, inputs in cases:
            with subtests.test(
                msg=f"copy_dim:{copy_dim}, k_val_shape:{tuple(inputs[1].shape)}"
            ):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(copy_dim=copy_dim),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class IndexPut(torch.nn.Module):
    def __init__(self, accumulate, in_place):
        super().__init__()
        self.accumulate = accumulate
        self.in_place = in_place

    def forward(self, x, indices, values):
        if self.in_place:
            result = x.clone()
            result.index_put_(indices, values, self.accumulate)
            return result
        return torch.index_put(x, indices, values, self.accumulate)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        sample_inputs = [
            # basic
            (torch.rand(5, 2), (torch.tensor([0, 2]),), torch.tensor([10.0, 20.0])),
            # shape
            (torch.rand(5), (torch.tensor([0, 2]),), torch.tensor([10.0, 20.0])),
            (
                torch.rand(5, 3, 2),
                (torch.tensor([0, 2]), torch.tensor([1, 1]), torch.tensor([0, 1])),
                torch.tensor([10.0, 20.0]),
            ),
            # indices
            (torch.rand(5, 2), (torch.tensor([2]),), torch.tensor([10.0])),
            (
                torch.rand(5, 3),
                (torch.tensor([0, 2, 4]),),
                torch.tensor([10.0, 20.0, 30.0]),
            ),
            (
                torch.rand(5),
                (torch.tensor([1, 1, 3, 3]),),
                torch.tensor([10.0, 20.0, 30.0, 40.0]),
            ),
            # broadcasting
            (torch.rand(5, 3), (torch.tensor([0, 2, 4]),), torch.tensor([42.0])),
            (
                torch.rand(3, 2, 2),
                (torch.tensor([0, 1]),),
                torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            ),
            (torch.rand(4, 2), (torch.tensor([1, 1, 1]),), torch.tensor([5.0])),
            # two-index
            (
                torch.rand(4, 3),
                (torch.tensor([0, 1, 2]), torch.tensor([1, 0, 2])),
                torch.tensor([10.0, 20.0, 30.0]),
            ),
            (
                torch.rand(3, 3),
                (torch.tensor([0, 2]), torch.tensor([1, 1])),
                torch.tensor([15.0, 25.0]),
            ),
            (
                torch.rand(3, 2),
                (torch.tensor([1, 1, 2]), torch.tensor([0, 0, 1])),
                torch.tensor([5.0, 10.0, 15.0]),
            ),
            (
                torch.rand(3, 2),
                (torch.tensor([1]), torch.tensor([0, 0, 1])),
                torch.tensor([5.0, 10.0, 15.0]),
            ),
        ]
        accumulates = [False, True]
        in_places = [False, True]
        for accumulate, in_place, inputs in itertools.product(
            accumulates, in_places, sample_inputs
        ):
            with subtests.test(
                msg=f"accumulate:{accumulate}, in_place:{in_place}, "
                f"x_shape:{tuple(inputs[0].shape)}"
            ):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(accumulate=accumulate, in_place=in_place),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class IndexSelect(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, indices):
        return torch.index_select(x, dim=self.dim, index=indices)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        x = torch.randn(2, 3, 4, 5)
        cases = [
            (0, torch.tensor([0, 1])),
            (1, torch.tensor([0, 2])),
            (2, torch.tensor([0, 2, 3])),
            (-1, torch.tensor([0, 2, 4])),
        ]
        for dim, indices in cases:
            with subtests.test(msg=f"dim:{dim}, n_indices:{indices.numel()}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim=dim),
                        inputs=(x, indices),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class InstanceNorm2d(torch.nn.Module):
    def __init__(self, n_features, affine):
        super().__init__()
        self.instance_norm = torch.nn.InstanceNorm2d(n_features, affine=affine)

    def forward(self, x):
        return self.instance_norm(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        n_features = 32
        inputs = (torch.randn([4, 32, 16, 16]),)
        for affine in [False, True]:
            with subtests.test(msg=f"affine:{affine}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(n_features=n_features, affine=affine),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Interpolate(torch.nn.Module):
    def __init__(self, scale_factor, align_corners, mode):
        super().__init__()
        self.align_corners = align_corners
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

    @staticmethod
    def _test(subtests, qnn_config, quantizer, compile_spec, expected, mode):
        scale_factors = [(2, 2)]
        align_corners = [True, False]
        inputs = (torch.randn(1, 2, 3, 4),)

        # test combo
        if mode in {"bilinear", "bicubic"}:
            for (
                scale_factor,
                align_corner,
            ) in itertools.product(scale_factors, align_corners):
                with subtests.test(
                    msg=f"scale_factor:{scale_factor}, align_corner:{align_corner}, mode:{mode}"
                ):
                    module = __class__(scale_factor, align_corner, mode)
                    with expected as metrics:
                        export_and_verify(
                            module=module,
                            inputs=inputs,
                            qnn_config=qnn_config,
                            quantizer=quantizer,
                            compile_specs=compile_spec,
                            metrics=metrics,
                        )
        else:
            # test nearest mode independently due to pytorch constraint
            for scale_factor in scale_factors:
                with subtests.test(msg=f"scale_factor:{scale_factor}, mode:{mode}"):
                    module = __class__(scale_factor, None, mode)
                    with expected as metrics:
                        export_and_verify(
                            module=module,
                            inputs=inputs,
                            qnn_config=qnn_config,
                            quantizer=quantizer,
                            compile_specs=compile_spec,
                            metrics=metrics,
                        )

    @staticmethod
    @unpack_fixtures
    def test_bilinear(subtests, qnn_config, quantizer, compile_spec, expected):
        __class__._test(
            subtests, qnn_config, quantizer, compile_spec, expected, "bilinear"
        )

    @staticmethod
    @unpack_fixtures
    def test_bicubic(subtests, qnn_config, quantizer, compile_spec, expected):
        __class__._test(
            subtests, qnn_config, quantizer, compile_spec, expected, "bicubic"
        )

    @staticmethod
    @unpack_fixtures
    def test_nearest(subtests, qnn_config, quantizer, compile_spec, expected):
        __class__._test(
            subtests, qnn_config, quantizer, compile_spec, expected, "nearest"
        )


class IsInf(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.isinf(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (
            torch.tensor(
                [
                    1.1,
                    float("inf"),
                    -float("inf"),
                    0.0,
                    float("nan"),
                    0.6,
                    float("nan"),
                    -5.0,
                ]
            ),
        )
        dtypes = [torch.float16, torch.float32]
        for dtype in dtypes:
            with subtests.test(msg=f"dtype:{dtype}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(),
                        inputs=tuple(i.to(dtype) for i in inputs),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class IsNan(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.isnan(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (
            torch.tensor(
                [
                    -2.0,
                    float("nan"),
                    -float("nan"),
                    0.2,
                    float("inf"),
                    3.2,
                    float("nan"),
                    -float("inf"),
                ]
            ),
        )
        dtypes = [torch.float16, torch.float32]
        for dtype in dtypes:
            with subtests.test(msg=f"dtype:{dtype}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(),
                        inputs=tuple(i.to(dtype) for i in inputs),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, elementwise_affine, bias, eps):
        super().__init__()
        self.layer_norm = torch.nn.LayerNorm(
            normalized_shape,
            eps=eps,
            elementwise_affine=elementwise_affine,
            bias=bias,
        )

    def forward(self, x):
        return self.layer_norm(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(196, 768),)

        # bias is only used when elementwise_affine=True in layer_norm
        # (normalized_shape, elementwise_affine, bias)
        cases = [
            ([768], False, False),
            ([768], True, True),
            ([768], True, False),
            ([196, 768], False, False),
        ]
        epsilons = [1e-6, 1e-2]
        for (normalized_shape, elementwise_affine, bias), eps in itertools.product(
            cases, epsilons
        ):
            with subtests.test(
                msg=f"normalized_shape:{normalized_shape}, elementwise_affine:{elementwise_affine}, bias:{bias}, eps:{eps}"
            ):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(
                            normalized_shape, elementwise_affine, bias, eps
                        ),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class LeakyReLU(torch.nn.Module):
    def __init__(self, negative_slope, inplace):
        super().__init__()
        self.leaky_relu = torch.nn.LeakyReLU(negative_slope, inplace=inplace)

    def forward(self, x):
        return self.leaky_relu(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(2, 5, 1, 3),)
        negative_slopes = [0.01, 0.05]
        inplaces = [False, True]
        for negative_slope, inplace in itertools.product(negative_slopes, inplaces):
            with subtests.test(
                msg=f"negative_slope:{negative_slope}, inplace:{inplace}"
            ):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(negative_slope, inplace),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class LessEqual(torch.nn.Module):
    def __init__(self, constant=None):
        super().__init__()
        self.constant = constant

    def forward(self, *args):
        if self.constant is not None:
            return args[0] <= self.constant
        return args[0] <= args[1]

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        with subtests.test(msg="(tensor, tensor)"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(),
                    inputs=(torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4)),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )
        with subtests.test(msg="(tensor, constant)"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(constant=0.5),
                    inputs=(torch.randn(1, 2, 3, 4),),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )


class LessThan(torch.nn.Module):
    def __init__(self, constant=None):
        super().__init__()
        self.constant = constant

    def forward(self, *args):
        if self.constant is not None:
            return self.constant < args[0]
        return args[0] < args[1]

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        with subtests.test(msg="(tensor, tensor)"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(),
                    inputs=(torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4)),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )
        with subtests.test(msg="(constant, tensor)"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(constant=0.5),
                    inputs=(torch.randn(1, 2, 3, 4),),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )


class LinalgVectorNorm(torch.nn.Module):
    def __init__(self, ord, dim, keepdim):
        super().__init__()
        self.ord = ord
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.linalg.vector_norm(
            x,
            ord=self.ord,
            dim=self.dim,
            keepdim=self.keepdim,
        )

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(3, 3),)
        ords = [1, 2, 3.5, float("inf"), float("-inf")]
        dims = [None, 1]
        keepdims = [False, True]
        for ord_val, dim, keepdim in itertools.product(ords, dims, keepdims):
            with subtests.test(msg=f"ord:{ord_val}, dim:{dim}, keepdim:{keepdim}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(ord=ord_val, dim=dim, keepdim=keepdim),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Linear(torch.nn.Module):
    def __init__(self, use_bias):
        super().__init__()
        self.linear = torch.nn.Linear(512, 32, use_bias)

    def forward(self, x):
        return self.linear(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(3, 512),)
        biases = [True, False]
        for use_bias in biases:
            with subtests.test(msg=f"use_bias:{use_bias}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(use_bias=use_bias),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class LinearNonConstantWeight(torch.nn.Module):
    def __init__(self, use_bias):
        super().__init__()
        self.input_dim = 512
        self.output_dim = 128
        self.linear = torch.nn.Linear(self.input_dim, 3 * self.output_dim, use_bias)

    def forward(self, x):
        w_q, w_k, w_v = self.linear.weight.split(
            [self.output_dim, self.output_dim, self.output_dim]
        )
        if self.linear.bias is not None:
            b_q, b_k, b_v = self.linear.bias.split(
                [self.output_dim, self.output_dim, self.output_dim]
            )
        else:
            b_q = b_k = b_v = None
        q = torch.nn.functional.linear(x, w_q, b_q)
        k = torch.nn.functional.linear(x, w_k, b_k)
        v = torch.nn.functional.linear(x, w_v, b_v)
        return q * k * v

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(3, 512),)
        biases = [True, False]
        for use_bias in biases:
            with subtests.test(msg=f"use_bias:{use_bias}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(use_bias=use_bias),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Log(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.rand(1, 2, 3, 4) + 0.1,)
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class Log10(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log10(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.abs(torch.rand(2, 5, 1, 3) + 0.1),)
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class Log1p(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log1p(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.abs(torch.rand(2, 5, 1, 3) + 0.1),)
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class Log2(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.log2(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.abs(torch.rand(2, 5, 1, 3) + 0.1),)
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class LogicalAnd(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.logical_and(x != 0, y != 0).float()

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (
            torch.tensor([0.0, 1.0, 10.0, 0.0]),
            torch.tensor([4.0, 0.0, 1.0, 0.0]),
        )
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class LogicalNot(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.logical_not(x > 0)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 3, 4),)
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class LogSoftmax(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.log_softmax(x, dim=self.dim)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 4, 8, 8),)
        dims = [-1, 1, 2]
        for dim in dims:
            with subtests.test(msg=f"dim:{dim}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class MaskedFill(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, attn_mask):
        return attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        attn_mask = torch.ones((64, 49, 49), dtype=torch.float32)
        for i in range(64):
            if i % 2 == 0:
                attn_mask[i, 35:, 35:] = 0
        inputs = (attn_mask,)
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class MaxDim(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, logits):
        max_logits, max_indices = torch.max(logits, dim=self.dim)
        return max_logits, max_indices

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(4, 10),)
        for dim in [-1, 0]:
            with subtests.test(msg=f"dim:{dim}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim=dim),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Maximum(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.maximum(x, y)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4))
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class MaxPool2d(torch.nn.Module):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        dilation,
        return_indices,
        ceil_mode,
    ):
        super().__init__()
        self.max_pool = torch.nn.MaxPool2d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

    def forward(self, x):
        return self.max_pool(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(4, 3, 24, 24),)
        kernel_sizes = [2, 3]
        strides = [1, 2]
        paddings = [0, 1]
        dilation = 1
        ceil_modes = [True, False]
        for kernel_size, stride, padding, ceil_mode in itertools.product(
            kernel_sizes, strides, paddings, ceil_modes
        ):
            with subtests.test(
                msg=f"kernel_size:{kernel_size}, stride:{stride}, "
                f"padding:{padding}, ceil_mode:{ceil_mode}"
            ):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(
                            kernel_size,
                            stride,
                            padding,
                            dilation,
                            return_indices=False,  # Aten turns this to true by default
                            ceil_mode=ceil_mode,
                        ),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class MaxPool3d(torch.nn.Module):
    def __init__(
        self,
        kernel_size,
        stride,
        padding,
        dilation,
        return_indices,
        ceil_mode,
    ):
        super().__init__()
        self.max_pool = torch.nn.MaxPool3d(
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            return_indices=return_indices,
            ceil_mode=ceil_mode,
        )

    def forward(self, x):
        return self.max_pool(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 7, 21, 35, 28),)
        kernel_sizes = [2, 3]
        strides = [1]
        paddings = [0, 1]
        dilation = 1
        ceil_modes = [True, False]

        for kernel_size, stride, padding, ceil_mode in itertools.product(
            kernel_sizes, strides, paddings, ceil_modes
        ):
            with subtests.test(
                msg=f"kernel_size:{kernel_size}, stride:{stride}, "
                f"padding:{padding}, ceil_mode:{ceil_mode}"
            ):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(
                            kernel_size,
                            stride,
                            padding,
                            dilation,
                            return_indices=False,  # Aten turns this to true by default
                            ceil_mode=ceil_mode,
                        ),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Mean(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        return torch.mean(x, dim=self.dim, keepdim=self.keepdim)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(2, 3, 4, 5),)
        dims = [None, [], 0, (-1, -2), (0, 2), (1, 3)]
        keepdims = [False, True]
        for dim, keepdim in itertools.product(dims, keepdims):
            with subtests.test(msg=f"dim:{dim}, keepdim:{keepdim}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim=dim, keepdim=keepdim),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class MinDim(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, logits):
        min_logits, min_indices = torch.min(logits, dim=self.dim)
        return min_logits, min_indices

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(4, 10),)
        for dim in [-1, 0]:
            with subtests.test(msg=f"dim:{dim}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim=dim),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Minimum(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.minimum(x, y)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4))
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


# TODO: Add more test cases based on real usage.
class MultiheadAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head_attention = torch.nn.MultiheadAttention(
            96, 12, dropout=0.0, batch_first=True
        )

    def forward(self, x):
        attn_output, _ = self.multi_head_attention(x, x, x, need_weights=False)
        return attn_output

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 197, 96),)
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class Narrow(torch.nn.Module):
    def __init__(self, dim, start, length):
        super().__init__()
        self.dim = dim
        self.start = start
        self.length = length

    def forward(self, x):
        return x.narrow(self.dim, self.start, self.length)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        # (input_shape, dim, start, length)
        cases = [
            ((1, 128, 64), 1, 4, 32),
            ((10, 20), 0, 2, 5),
            ((2, 8, 16, 32), 2, 2, 8),
            ((1, 4, 8, 16), 3, 0, 4),
        ]
        for input_shape, dim, start, length in cases:
            with subtests.test(
                msg=f"shape:{input_shape}, dim:{dim}, start:{start}, length:{length}"
            ):
                inputs = (torch.randn(*input_shape),)
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim, start, length),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Neg(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.neg(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 4, 16, 16),)
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class NotEqual(torch.nn.Module):
    def __init__(self, constant=None):
        super().__init__()
        self.constant = constant

    def forward(self, *args):
        if self.constant is not None:
            return args[0] != self.constant
        return args[0] != args[1]

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        with subtests.test(msg="(tensor, tensor)"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(),
                    inputs=(torch.randn(1, 2, 3, 4), torch.randn(2, 3, 4)),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )
        with subtests.test(msg="(tensor, constant)"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(constant=0.5),
                    inputs=(torch.randn(1, 2, 3, 4),),
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )


class Pad(torch.nn.Module):
    def __init__(self, padding, mode, value=0.0):
        super().__init__()
        self.padding = padding
        self.mode = mode
        self.value = value

    def forward(self, x):
        kwargs = {"mode": self.mode}
        if self.mode == "constant":
            kwargs["value"] = self.value
        return torch.nn.functional.pad(x, self.padding, **kwargs)

    @staticmethod
    @unpack_fixtures
    def test_constant(subtests, qnn_config, quantizer, compile_spec, expected):
        # name, padding, input shape, value
        cases = [
            ("2d_trailing_only_non_zero", [1, 1], (4, 8), 4.5),
            ("2d_all_dims", [1, 1, 2, 1], (4, 8), 0.0),
            ("3d_partial_dim", [0, 4, 0, 1, 0, 0], (2, 4, 8), 0),
            ("3d_all_dims_nonzero", [1, 1, 1, 2, 2, 1], (2, 4, 8), 2.5),
            ("4d_nonzero", [1, 1, 1, 1], (1, 4, 8, 8), 1.5),
            ("4d_all_dims", [1, 2, 1, 2, 1, 2, 1, 2], (1, 4, 8, 8), 0.0),
            ("4d_partial_dims_non_zero", [0, 1, 0, 2, 1, 3, 1, 2], (1, 4, 8, 8), 1.5),
        ]
        for case_name, padding, shape, value in cases:
            with subtests.test(msg=case_name):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(padding=padding, mode="constant", value=value),
                        inputs=(torch.randn(*shape),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )

    @staticmethod
    @unpack_fixtures
    def test_reflect(subtests, qnn_config, quantizer, compile_spec, expected):
        cases = [
            ("2d_pad_partial", [0, 1], (4, 8)),
            ("2d_pad", [2, 3], (4, 8)),
            ("3d_pad_partial", [0, 0, 1, 1], (1, 4, 8)),
            ("3d_pad", [2, 1, 1, 2], (1, 4, 8)),
        ]
        for name, padding, shape in cases:
            with subtests.test(msg=name):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(padding=padding, mode="reflect"),
                        inputs=(torch.randn(*shape),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Permute(torch.nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(2, 3, 4, 5),)
        for dims in [[0, 2, 3, 1], [-1, -3, -2, -4]]:
            with subtests.test(msg=f"dims:{dims}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dims=dims),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class PixelShuffle(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.pixel_shuffle = torch.nn.PixelShuffle(scale)

    def forward(self, x):
        return self.pixel_shuffle(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(2, 4, 3, 3),)
        scale = 2
        with expected as metrics:
            export_and_verify(
                module=__class__(scale=scale),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class PixelUnshuffle(torch.nn.Module):
    def __init__(self, downscale_factor):
        super().__init__()
        self.pixel_unshuffle = torch.nn.PixelUnshuffle(downscale_factor)

    def forward(self, x):
        return self.pixel_unshuffle(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(2, 2, 6, 6),)
        downscale_factor = 2
        with expected as metrics:
            export_and_verify(
                module=__class__(downscale_factor=downscale_factor),
                inputs=inputs,
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class PowScalar(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, x):
        return torch.ops.aten.pow.Scalar(self.base, x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.rand(10, 10) + 0.1,)
        bases = [2.0, 3.0, 9, 0.5]
        for base in bases:
            with subtests.test(msg=f"base:{base}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(base=base),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class PowTensorScalar(torch.nn.Module):
    def __init__(self, exponent):
        super().__init__()
        self.exponent = exponent

    def forward(self, x):
        return torch.pow(x, self.exponent)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        default_inputs = (torch.rand(10, 10),)
        for exponent in [2, 1, -1, -0.5, 0.5, 10]:
            with subtests.test(msg=f"exponent:{exponent}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(exponent=exponent),
                        inputs=default_inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class PReLU(torch.nn.Module):
    def __init__(self, num_parameters, init):
        super().__init__()
        self.prelu = torch.nn.PReLU(num_parameters=num_parameters, init=init)

    def forward(self, x):
        return self.prelu(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(2, 5, 1, 3),)
        num_parameters_options = [1, 5]
        init_options = [0.1, 0.25, 0.5]
        for num_parameters, init in itertools.product(
            num_parameters_options, init_options
        ):
            with subtests.test(msg=f"num_parameters:{num_parameters}, init:{init}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(num_parameters=num_parameters, init=init),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Rand(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.rand_like(x) + x

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        for shape in [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]:
            with subtests.test(msg=str(shape)):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(),
                        inputs=(torch.randn(*shape),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Reciprocal(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.reciprocal(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        for shape in [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5), (2, 3, 4, 5, 6)]:
            with subtests.test(msg=str(shape)):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(),
                        inputs=(torch.randn(*shape),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class ReflectionPad(torch.nn.Module):
    def __init__(self, ndim, padding):
        super().__init__()
        self.pad = getattr(torch.nn, f"ReflectionPad{ndim}d")(padding)

    def forward(self, x):
        return self.pad(x)

    @staticmethod
    @unpack_fixtures
    def test_3d(qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 3, 8),)
        for padding in [2, (1, 3)]:
            with expected as metrics:
                export_and_verify(
                    module=__class__(ndim=1, padding=padding),
                    inputs=inputs,
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )

    @staticmethod
    @unpack_fixtures
    def test_4d(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 3, 8, 8),)
        for padding in [2, (1, 2, 1, 2)]:
            with subtests.test(msg=f"padding={padding}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(2, padding),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Relu(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        return self.relu(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        for shape in [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5), (2, 3, 4, 5, 6)]:
            with subtests.test(msg=str(shape)):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(),
                        inputs=(torch.randn(*shape),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Relu6(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.relu6 = torch.nn.ReLU6()

    def forward(self, x):
        return self.relu6(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        # 5D is not supported
        for shape in [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]:
            with subtests.test(msg=str(shape)):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(),
                        inputs=(torch.randn(*shape),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Remainder(torch.nn.Module):
    class _Scalar(torch.nn.Module):
        def __init__(self, scalar):
            super().__init__()
            self.scalar = scalar

        def forward(self, x):
            return torch.remainder(x, self.scalar)

    class _Tensor(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            return torch.remainder(x, y)

    class _ScalarTensor(torch.nn.Module):
        def __init__(self, scalar):
            super().__init__()
            self.scalar = scalar

        def forward(self, x):
            return torch.remainder(self.scalar, x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        # 5D is not supported
        inputs = [
            (torch.arange(-1.5, 1.5, 1.0).reshape(3),),
            (torch.arange(-4.5, 4.5, 1.0).reshape(3, 3),),
            (torch.arange(-13.5, 13.5, 1.0).reshape(3, 3, 3),),
            (torch.arange(-40.5, 40.5, 1.0).reshape(3, 3, 3, 3),),
        ]
        scalars = [0.5, 3.0]
        for scalar, inp in itertools.product(scalars, inputs):
            modules = [
                Remainder._Scalar(scalar),
                Remainder._ScalarTensor(scalar),
            ]
            for module in modules:
                with subtests.test(
                    msg=f"{type(module).__name__} with input shape {str(inp[0].shape)}, scalar: {scalar}"
                ):
                    with expected as metrics:
                        export_and_verify(
                            module=module,
                            inputs=inp,
                            qnn_config=qnn_config,
                            quantizer=quantizer,
                            compile_specs=compile_spec,
                            metrics=metrics,
                        )
        for inp in inputs:
            inp2 = inp[0] + 1.0
            module = Remainder._Tensor()
            with subtests.test(
                msg=f"{type(module).__name__} with input shape {str(inp[0].shape)}"
            ):
                with expected as metrics:
                    export_and_verify(
                        module=module,
                        inputs=(inp[0], inp2),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Repeat(torch.nn.Module):
    def __init__(self, repeats):
        super().__init__()
        self.repeats = repeats

    def forward(self, x):
        return x.repeat(*self.repeats)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        for shape in [(2,), (2, 2), (2, 2, 2), (2, 2, 2, 2), (2, 2, 2, 2, 2)]:
            inputs = (torch.randn(*shape),)
            with subtests.test(msg=str(shape)):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(repeats=shape),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Reshape(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.reshape(*self.shape)

    @staticmethod
    @unpack_fixtures
    def test_2d_to_4d_random_reshape(
        subtests, qnn_config, quantizer, compile_spec, expected
    ):
        for shape in [
            (2, 3),
            (2, 3, 4),
            (2, 3, 4, 5),
        ]:
            target_shape = random.sample(shape, k=len(shape))
            inputs = (torch.randn(*shape),)
            with subtests.test(msg=f"reshape {str(shape)} to {str(target_shape)}"):
                with expected as metrics:
                    export_and_verify(
                        # reverse the shape as target shape
                        module=__class__(target_shape),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )

    @staticmethod
    @unpack_fixtures
    def test_2d_to_4d_flatten_last_two_dims(
        subtests, qnn_config, quantizer, compile_spec, expected
    ):
        for shape in [
            (2, 3),
            (2, 3, 4),
            (2, 3, 4, 5),
        ]:
            target_shape = (*shape[:-2], -1)
            inputs = (torch.randn(*shape),)
            with subtests.test(msg=f"reshape {str(shape)} to {str(target_shape)}"):
                with expected as metrics:
                    export_and_verify(
                        # flatten last two dims
                        module=__class__(target_shape),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )

    @staticmethod
    @unpack_fixtures
    def test_5d_random_reshape(subtests, qnn_config, quantizer, compile_spec, expected):
        shape = (2, 3, 4, 5, 6)
        target_shape = random.sample(shape, k=len(shape))
        inputs = (torch.randn(*shape),)
        with subtests.test(msg=f"reshape {str(shape)} to {str(target_shape)}"):
            with expected as metrics:
                export_and_verify(
                    # reverse the shape as target shape
                    module=__class__(target_shape),
                    inputs=inputs,
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )

    @staticmethod
    @unpack_fixtures
    def test_5d_flatten_last_two_dims(
        subtests, qnn_config, quantizer, compile_spec, expected
    ):
        shape = (2, 3, 4, 5, 6)
        target_shape = (*shape[:-2], -1)
        inputs = (torch.randn(*shape),)
        with subtests.test(msg=f"reshape {str(shape)} to {str(target_shape)}"):
            with expected as metrics:
                export_and_verify(
                    # flatten last two
                    module=__class__(target_shape),
                    inputs=inputs,
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )


class RmsNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps, elementwise_affine):
        super().__init__()
        self.rms = torch.nn.RMSNorm(
            normalized_shape, eps=eps, elementwise_affine=elementwise_affine
        )

    def forward(self, x):
        return self.rms(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        # HTP only supports normalization on channel
        for shape in [(2, 3), (2, 3, 4), (2, 3, 4, 5)]:
            normalized_shape = shape[-1]
            for eps, elementwise_affine in itertools.product(
                [None, 1e-5, 1e-4], [True, False]
            ):
                inputs = (torch.randn(*shape),)
                cfg = {
                    "normalized_shape": normalized_shape,
                    "eps": eps,
                    "elementwise_affine": elementwise_affine,
                }
                with subtests.test(msg=f"input shape {shape} with {str(cfg)}"):
                    with expected as metrics:
                        export_and_verify(
                            module=__class__(**cfg),
                            inputs=inputs,
                            qnn_config=qnn_config,
                            quantizer=quantizer,
                            compile_specs=compile_spec,
                            metrics=metrics,
                        )


class Roll(torch.nn.Module):
    def __init__(self, shifts, dims):
        super().__init__()
        self.shifts = shifts
        self.dims = dims

    def forward(self, x):
        return torch.roll(x, shifts=self.shifts, dims=self.dims)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 56, 56, 96),)
        configs = [
            {"shifts": (3, 3), "dims": (1, 2)},
            {"shifts": (70, 59), "dims": (1, 2)},
            {"shifts": (3, 56), "dims": (1, 2)},
            {"shifts": 3, "dims": None},
            {"shifts": -3, "dims": None},
            {"shifts": 0, "dims": None},
            {"shifts": 56 * 56 * 96 + 3, "dims": None},
        ]
        for cfg in configs:
            with subtests.test(msg=str(cfg)):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(**cfg),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Round(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.round(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        # 5D is not supported
        shapes = [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]
        for shape in shapes:
            inputs = (torch.randn(*shape),)
            with subtests.test(msg=f"shape:{shape}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Rsqrt(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.rsqrt(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        for shape in [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5), (2, 3, 4, 5, 6)]:
            inputs = (torch.randn(*shape).abs() + 1e-3,)
            with subtests.test(msg=str(shape)):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class ScatterSrc(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, data, index, src):
        return torch.scatter(data, self.dim, index, src)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = [
            (
                torch.zeros(3, 5),
                torch.tensor(
                    [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [1, 0, 3, 4, 2]],
                    dtype=torch.int64,
                ),
                torch.randn(3, 5),
            ),
        ]
        for dim, (data, index, src) in itertools.product([-1, 1], inputs):
            with subtests.test(msg=f"dim={dim}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim=dim),
                        inputs=(data, index, src),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class ScaledDotProductAttention(torch.nn.Module):
    def __init__(self, scale, is_causal, use_mask):
        super().__init__()
        self.scale = scale
        self.is_causal = is_causal
        self.use_mask = use_mask

    def forward(self, *args):
        if self.use_mask:
            query_layer, key_layer, value_layer, attn_mask = args
            return torch.nn.functional.scaled_dot_product_attention(
                query_layer,
                key_layer,
                value_layer,
                attn_mask,
                is_causal=self.is_causal,
                scale=self.scale,
            )
        query_layer, key_layer, value_layer = args
        return torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            is_causal=self.is_causal,
            scale=self.scale,
        )

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        mask = torch.tril(torch.randn(1, 1, 100, 100))
        mask[mask == 0] = float("-inf")
        qkv = (
            torch.randn(1, 4, 100, 64),
            torch.randn(1, 4, 100, 64),
            torch.randn(1, 4, 100, 64),
        )
        masked_inputs = (*qkv, mask)
        # explicit-mask cases: scale sweep, is_causal must remain False
        for scale in [None, 0.5]:
            with subtests.test(msg=f"masked, scale:{scale}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(scale=scale, is_causal=False, use_mask=True),
                        inputs=masked_inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )
        # no-mask cases
        for scale, is_causal in itertools.product([None, 0.5], [True, False]):
            with subtests.test(msg=f"is_causal:{is_causal}, scale:{scale}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(
                            scale=scale, is_causal=is_causal, use_mask=False
                        ),
                        inputs=qkv,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class SelectScatter(torch.nn.Module):
    def __init__(self, dim, index):
        super().__init__()
        self.dim = dim
        self.index = index

    def forward(self, x, y):
        return x.select_scatter(y, dim=self.dim, index=self.index)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        cases = [
            (0, 2, (torch.randn(4, 8), torch.randn(8))),
            (1, 0, (torch.randn(3, 4, 5), torch.randn(3, 5))),
            (1, -1, (torch.randn(3, 4, 5), torch.randn(3, 5))),
            (-1, 2, (torch.randn(3, 4, 5), torch.randn(3, 4))),
            (3, 1, (torch.randn(2, 3, 4, 5), torch.randn(2, 3, 4))),
        ]
        for dim, index, inputs in cases:
            with subtests.test(msg=f"dim={dim}_index={index}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim=dim, index=index),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class SelectCopy(torch.nn.Module):
    def __init__(self, dim, index):
        super().__init__()
        self.dim = dim
        self.index = index

    def forward(self, x):
        return x.select(dim=self.dim, index=self.index)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 3, 3, 3),)
        configs = [
            {"dim": 0, "index": 0},
            {"dim": 1, "index": 1},
            {"dim": -1, "index": 0},
            {"dim": 2, "index": 1},
        ]
        for cfg in configs:
            with subtests.test(msg=str(cfg)):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(**cfg),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Sigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        for shape in [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5), (2, 3, 4, 5, 6)]:
            with subtests.test(msg=str(shape)):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(),
                        inputs=(torch.randn(*shape),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Sign(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sign(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        # 5D is not supported
        for shape in [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]:
            with subtests.test(msg=str(shape)):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(),
                        inputs=(torch.randn(*shape),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Sin(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        # 5D is not supported
        for shape in [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]:
            with subtests.test(msg=str(shape)):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(),
                        inputs=(torch.randn(*shape),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class SliceCopy(torch.nn.Module):
    class _Default(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.cat([x[:1], x[1:]], dim=1)

    class _BufferSlice(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.position_ids = torch.randn([1, 512])

        def forward(self, x, y):
            seq_length = y.size()[1]
            return x[:, :seq_length] + self.position_ids[:, :seq_length]

    class _BufferSliceWithStep(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.position_ids = torch.randn([1, 512])
            self.step = 2

        def forward(self, x, y):
            seq_length = y.size()[1]
            return (
                x[:, : seq_length : self.step]
                + self.position_ids[:, : seq_length : self.step]
            )

    class _NegStart(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[:, -4:]

    class _FullSlice(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x[:, :] + 1.0

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        cases = [
            (SliceCopy._Default(), (torch.randn(2, 1, 320, 512),)),
            (SliceCopy._BufferSlice(), (torch.randn(1, 512), torch.randn(1, 8))),
            (
                SliceCopy._BufferSliceWithStep(),
                (torch.randn(1, 512), torch.randn(1, 8)),
            ),
            (SliceCopy._NegStart(), (torch.randn(2, 8),)),
            (SliceCopy._FullSlice(), (torch.randn(2, 8),)),
        ]
        for module, inputs in cases:
            with subtests.test(msg=type(module).__name__):
                with expected as metrics:
                    export_and_verify(
                        module=module,
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class SliceScatter(torch.nn.Module):
    def __init__(self, dim, start, end, step):
        super().__init__()
        self.dim = dim
        self.start = start
        self.end = end
        self.step = step

    def forward(self, x, y):
        return x.slice_scatter(
            y, dim=self.dim, start=self.start, end=self.end, step=self.step
        )

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        cases = [
            (
                __class__(dim=0, start=3, end=5, step=1),
                (torch.zeros(8, 8), torch.ones(2, 8)),
            ),
            (
                __class__(dim=1, start=2, end=6, step=2),
                (torch.zeros(8, 8), torch.ones(8, 2)),
            ),
            (
                __class__(dim=2, start=1, end=4, step=1),
                (torch.zeros(2, 4, 6), torch.ones(2, 4, 3)),
            ),
            (
                __class__(dim=-1, start=0, end=4, step=1),
                (torch.zeros(1, 2, 3, 8), torch.ones(1, 2, 3, 4)),
            ),
        ]
        for module, inputs in cases:
            with subtests.test(msg=f"dim:{module.dim}"):
                with expected as metrics:
                    export_and_verify(
                        module=module,
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Softmax(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.nn.functional.softmax(x, dim=self.dim)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        shapes = [(2, 3), (2, 3, 4), (1, 4, 8, 8), (1, 2, 3, 4, 5)]
        # QNN HTP Softmax only support axis=-1,
        # tricky layout transform logic is added to suppose axis=1
        dims = [-1, 1]
        for shape, dim in itertools.product(shapes, dims):
            with subtests.test(msg=f"shape:{shape}, dim:{dim}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim),
                        inputs=(torch.randn(*shape),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Split(torch.nn.Module):
    def __init__(self, split_size_or_sections, dim):
        super().__init__()
        self.split_size_or_sections = split_size_or_sections
        self.dim = dim

    def forward(self, x):
        return torch.split(
            tensor=x,
            split_size_or_sections=self.split_size_or_sections,
            dim=self.dim,
        )

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 3, 4),)
        dims = [3, -1]
        split_size_or_sections = [1, [1, 2, 1]]
        for dim, split_size_or_section in itertools.product(
            dims, split_size_or_sections
        ):
            with subtests.test(
                msg=f"dim:{dim}, split_size_or_sections:{split_size_or_section}"
            ):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(
                            split_size_or_sections=split_size_or_section,
                            dim=dim,
                        ),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Square(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.square(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        for shape in [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5), (2, 3, 4, 5, 6)]:
            with subtests.test(msg=str(shape)):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(),
                        inputs=(torch.randn(*shape),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Squeeze(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if self.dim is None:
            return x.squeeze()
        return x.squeeze(self.dim)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        cases = [
            # squeeze(Tensor)
            (None, (1, 3, 3)),
            (None, (1, 3, 1, 4)),
            (None, (2, 1, 3, 1, 4)),
            # squeeze.dim(Tensor, int)
            (0, (1, 3, 3)),
            (-1, (1, 3, 1)),
            (2, (1, 3, 1, 4)),
            # squeeze.dims(Tensor, int[])
            ([0, 2], (1, 3, 1, 4)),
            ([1, 3], (2, 1, 3, 1, 4)),
        ]
        for dim, shape in cases:
            with subtests.test(msg=f"dim:{dim}, shape:{shape}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim=dim),
                        inputs=(torch.randn(*shape),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Stack(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, y, z):
        return torch.stack((x, y, z), dim=self.dim)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (
            torch.randn(1, 2, 3, 4),
            torch.randn(1, 2, 3, 4),
            torch.randn(1, 2, 3, 4),
        )
        for dim in [0, 1, -1]:
            with subtests.test(msg=f"dim:{dim}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim=dim),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class SumIntList(torch.nn.Module):
    def __init__(self, dim, keepdim):
        super().__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, x):
        if self.dim is None:
            return torch.sum(x)
        return torch.sum(x, dim=self.dim, keepdim=self.keepdim)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 4, 8, 8),)
        dims = [(2, 3), (-1,), 1]
        keepdims = [False, True]
        for dim, keepdim in itertools.product(dims, keepdims):
            with subtests.test(msg=f"dim:{dim}, keepdim:{keepdim}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim=dim, keepdim=keepdim),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class SwapAxes(torch.nn.Module):
    def __init__(self, axis0, axis1):
        super().__init__()
        self.axis0 = axis0
        self.axis1 = axis1

    def forward(self, x):
        return torch.swapaxes(x, axis0=self.axis0, axis1=self.axis1)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 3, 4),)
        for axis0, axis1 in [(0, 1), (1, 2), (-1, -2), (2, 3)]:
            with subtests.test(msg=f"axis0:{axis0}, axis1:{axis1}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(axis0, axis1),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Tan(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tan(x)

    @staticmethod
    @unpack_fixtures
    def test(qnn_config, quantizer, compile_spec, expected):
        with expected as metrics:
            export_and_verify(
                module=__class__(),
                inputs=(torch.randn(2, 5, 1, 3),),
                qnn_config=qnn_config,
                quantizer=quantizer,
                compile_specs=compile_spec,
                metrics=metrics,
            )


class Tanh(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tanh(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        # 5D is not supported
        for shape in [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]:
            with subtests.test(msg=str(shape)):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(),
                        inputs=(torch.randn(*shape),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Threshold(torch.nn.Module):
    def __init__(self, threshold, value, inplace):
        super().__init__()
        self.threshold = threshold
        self.value = value
        self.inplace = inplace

    def forward(self, x):
        return torch.nn.functional.threshold(
            x, threshold=self.threshold, value=self.value, inplace=self.inplace
        )

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(2, 5, 1, 3),)
        for threshold, value, inplace in itertools.product(
            [0.0, 0.5, -0.5], [0.0, 3.0, -1.0], [True, False]
        ):
            cfg = {"threshold": threshold, "value": value, "inplace": inplace}
            with subtests.test(msg=str(cfg)):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(**cfg),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class TopK(torch.nn.Module):
    def __init__(self, k, dim, largest):
        super().__init__()
        self.k = k
        self.dim = dim
        self.largest = largest

    def forward(self, x):
        topk, indices = torch.topk(x, k=self.k, dim=self.dim, largest=self.largest)
        return topk + torch.gather(x, dim=-1, index=indices)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 2, 3, 4),)
        ks = [1, 3]
        # TODO: extend this once QNN starts to support more dimension
        dims = [-1]
        largests = [True, False]
        for k, dim, largest in itertools.product(ks, dims, largests):
            with subtests.test(msg=f"k:{k}, dim:{dim}, largest:{largest}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(k=k, dim=dim, largest=largest),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Triu(torch.nn.Module):
    def __init__(self, diagonal):
        super().__init__()
        self.diagonal = diagonal

    def forward(self, x):
        if self.diagonal is not None:
            return torch.triu(x, diagonal=self.diagonal)
        return torch.triu(x)

    class _Constant(torch.nn.Module):
        def __init__(self, diagonal, constant_dtype):
            super().__init__()
            self.diagonal = diagonal
            self.constant_dtype = constant_dtype
            self.register_buffer("mask", torch.ones((5, 5), dtype=constant_dtype))

        def forward(self, x):
            mask = torch.triu(self.mask, diagonal=self.diagonal)
            if self.constant_dtype == torch.bool:
                mask = torch.zeros(x.shape, dtype=x.dtype).masked_fill_(mask, -10000.0)
            return mask + x

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        cases = [
            (Triu(diagonal=None), (torch.randn(3, 3),)),
            (Triu(diagonal=None), (torch.randn(1, 2, 3, 3),)),
            (Triu(diagonal=1), (torch.randn(3, 3),)),
            (Triu(diagonal=1), (torch.randn(1, 2, 3, 3),)),
            (Triu(diagonal=-1), (torch.randn(4, 5),)),
            (Triu(diagonal=2), (torch.randn(5, 5),)),
        ]
        for module, inputs in cases:
            with subtests.test(msg=f"{type(module).__name__}_{inputs[0].shape}"):
                with expected as metrics:
                    export_and_verify(
                        module=module,
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )

    @staticmethod
    @unpack_fixtures
    def test_constant(subtests, qnn_config, quantizer, compile_spec, expected):
        cases = [
            (Triu._Constant(1, constant_dtype=torch.bool), (torch.zeros(5, 5),)),
            (Triu._Constant(1, constant_dtype=torch.float32), (torch.zeros(5, 5),)),
            (Triu._Constant(-1, constant_dtype=torch.float32), (torch.zeros(5, 5),)),
        ]
        for module, inputs in cases:
            with subtests.test(msg=f"{type(module).__name__}_{inputs[0].shape}"):
                with expected as metrics:
                    export_and_verify(
                        module=module,
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Trunc(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.trunc(x)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        # 5D is not supported
        for shape in [(2,), (2, 3), (2, 3, 4), (2, 3, 4, 5)]:
            with subtests.test(msg=str(shape)):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(),
                        inputs=(torch.randn(*shape),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Unbind(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.unbind(x, dim=self.dim)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        cases = [
            ((3, 3), 0),
            ((3, 3), 1),
            ((3, 3), -1),
            ((1, 2, 3, 4), 0),
            ((1, 2, 3, 4), 2),
            ((1, 2, 3, 4), -1),
        ]
        for shape, dim in cases:
            with subtests.test(msg=f"shape:{shape}, dim:{dim}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim=dim),
                        inputs=(torch.randn(*shape),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Unflatten(torch.nn.Module):
    def __init__(self, dim, sizes):
        super().__init__()
        self.dim = dim
        self.sizes = sizes

    def forward(self, x):
        return torch.unflatten(x, dim=self.dim, sizes=self.sizes)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(1, 24),)
        configs = [
            {"dim": 1, "sizes": (2, 3, 4)},
            {"dim": 1, "sizes": (4, 6)},
            {"dim": 1, "sizes": (2, -1, 4)},
            {"dim": -1, "sizes": (4, 6)},
        ]
        for cfg in configs:
            with subtests.test(msg=str(cfg)):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(**cfg),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Unfold(torch.nn.Module):
    def __init__(self, kernel_size, stride, padding, dilation):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

    def forward(self, x):
        return torch.nn.functional.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        inputs = (torch.randn(2, 128, 32, 32),)
        configs = [
            {"kernel_size": 2, "stride": 2, "padding": 0, "dilation": 1},
            {"kernel_size": 4, "stride": 4, "padding": 0, "dilation": 1},
            {"kernel_size": (2, 2), "stride": (2, 2), "padding": 0, "dilation": 1},
            {"kernel_size": (4, 4), "stride": (4, 4), "padding": 0, "dilation": 1},
        ]
        for cfg in configs:
            with subtests.test(msg=str(cfg)):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(**cfg),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )

    @staticmethod
    @unpack_fixtures
    def test_unsupported(subtests, qnn_config, quantizer, compile_spec, expected):
        # stride != kernel_size is not supported by DecomposeColIm
        inputs = (torch.randn(2, 128, 32, 32),)
        configs = [
            {"kernel_size": (2, 2), "stride": (2, 1), "padding": 0, "dilation": 1},
            {"kernel_size": (4, 4), "stride": (2, 2), "padding": 0, "dilation": 1},
            {"kernel_size": (2, 2), "stride": (2, 2), "dilation": (2, 2)},
            {"kernel_size": (2, 2), "stride": (2, 2), "padding": (1, 1)},
        ]
        for cfg in configs:
            with subtests.test(msg=str(cfg)):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(**cfg),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )
        inputs = (torch.randn(2, 128, 32),)
        cfg = {"kernel_size": (2, 2), "stride": (2, 2)}
        with subtests.test(msg="Unsupported input shape"):
            with expected as metrics:
                export_and_verify(
                    module=__class__(**cfg),
                    inputs=inputs,
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )


class Unsqueeze(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        cases = [
            ((1, 3, 3), 0),
            ((1, 3, 3), 1),
            ((1, 3, 3), -1),
            ((1, 2, 3, 4), 0),
            ((1, 2, 3, 4), -1),
            ((1, 2, 3, 4), -2),
        ]
        for shape, dim in cases:
            with subtests.test(msg=f"shape:{shape}, dim:{dim}"):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(dim=dim),
                        inputs=(torch.randn(*shape),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class View(torch.nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)

    @staticmethod
    @unpack_fixtures
    def test_2d_to_4d_random_reshape(
        subtests, qnn_config, quantizer, compile_spec, expected
    ):
        for shape in [
            (2, 3),
            (2, 3, 4),
            (2, 3, 4, 5),
        ]:
            target_shape = random.sample(shape, k=len(shape))
            inputs = (torch.randn(*shape),)
            with subtests.test(msg=f"reshape {str(shape)} to {str(target_shape)}"):
                with expected as metrics:
                    export_and_verify(
                        # reverse the shape as target shape
                        module=__class__(target_shape),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )

    @staticmethod
    @unpack_fixtures
    def test_2d_to_4d_flatten_last_two_dims(
        subtests, qnn_config, quantizer, compile_spec, expected
    ):
        for shape in [
            (2, 3),
            (2, 3, 4),
            (2, 3, 4, 5),
        ]:
            target_shape = (*shape[:-2], -1)
            inputs = (torch.randn(*shape),)
            with subtests.test(msg=f"reshape {str(shape)} to {str(target_shape)}"):
                with expected as metrics:
                    export_and_verify(
                        # flatten last two dims
                        module=__class__(target_shape),
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )

    @staticmethod
    @unpack_fixtures
    def test_5d_random_reshape(subtests, qnn_config, quantizer, compile_spec, expected):
        shape = (2, 3, 4, 5, 6)
        target_shape = random.sample(shape, k=len(shape))
        inputs = (torch.randn(*shape),)
        with subtests.test(msg=f"reshape {str(shape)} to {str(target_shape)}"):
            with expected as metrics:
                export_and_verify(
                    # reverse the shape as target shape
                    module=__class__(target_shape),
                    inputs=inputs,
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )

    @staticmethod
    @unpack_fixtures
    def test_5d_flatten_last_two_dims(
        subtests, qnn_config, quantizer, compile_spec, expected
    ):
        shape = (2, 3, 4, 5, 6)
        target_shape = (*shape[:-2], -1)
        inputs = (torch.randn(*shape),)
        with subtests.test(msg=f"reshape {str(shape)} to {str(target_shape)}"):
            with expected as metrics:
                export_and_verify(
                    # flatten last two
                    module=__class__(target_shape),
                    inputs=inputs,
                    qnn_config=qnn_config,
                    quantizer=quantizer,
                    compile_specs=compile_spec,
                    metrics=metrics,
                )


class Where(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y, z):
        return torch.where(x >= torch.zeros(x.shape), y, z)

    class _Constant(torch.nn.Module):
        def __init__(self, pos, neg):
            super().__init__()
            self.register_buffer("pos", pos)
            self.register_buffer("neg", neg)

        def forward(self, x):
            return torch.where(x >= torch.zeros(x.shape), self.pos, self.neg)

    class _ConstantOther(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.where(x >= 0, torch.ones(x.shape), 0)

    class _ConstantInf(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return torch.nn.functional.softmax(
                torch.where(x >= 0, 0.1, float("-inf")), dim=-1
            )

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        cases = [
            (
                Where(),
                (torch.randn(3, 2), torch.randn(3, 2), torch.randn(3, 2)),
            ),
            (
                Where(),
                (
                    torch.randn(1, 2, 3, 4),
                    torch.randn(1, 2, 3, 4),
                    torch.randn(1, 2, 3, 4),
                ),
            ),
            (
                Where._Constant(torch.randn(3, 2), torch.randn(3, 2)),
                (torch.randn(3, 2),),
            ),
            (
                Where._Constant(torch.randn(1, 2, 3, 4), torch.randn(1, 2, 3, 4)),
                (torch.randn(1, 2, 3, 4),),
            ),
            (
                Where._ConstantOther(),
                (torch.randn(3, 2),),
            ),
            (
                Where._ConstantInf(),
                (torch.randn(30, 20),),
            ),
        ]
        for module, inputs in cases:
            with subtests.test(msg=f"{type(module).__name__}_{inputs[0].shape}"):
                with expected as metrics:
                    export_and_verify(
                        module=module,
                        inputs=inputs,
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )


class Var(torch.nn.Module):
    def __init__(self, dim, correction, keepdim):
        super().__init__()
        self.dim = dim
        self.correction = correction
        self.keepdim = keepdim

    def forward(self, x):
        return torch.var(
            x, dim=self.dim, correction=self.correction, keepdim=self.keepdim
        )

    @staticmethod
    @unpack_fixtures
    def test(subtests, qnn_config, quantizer, compile_spec, expected):
        dims = [-1, [0, 2]]
        corrections = [0, 1]
        keepdims = [False, True]
        for dim, correction, keepdim in itertools.product(dims, corrections, keepdims):
            with subtests.test(
                msg=f"dim:{dim}, correction:{correction}, keepdim:{keepdim}"
            ):
                with expected as metrics:
                    export_and_verify(
                        module=__class__(
                            dim=dim, correction=correction, keepdim=keepdim
                        ),
                        inputs=(torch.randn(3, 4, 5),),
                        qnn_config=qnn_config,
                        quantizer=quantizer,
                        compile_specs=compile_spec,
                        metrics=metrics,
                    )
