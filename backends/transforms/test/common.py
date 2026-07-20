# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, ParamSpec, TypeVar

import pytest

xfail_type = str | tuple[str, type[Exception]]
_P = ParamSpec("_P")
_R = TypeVar("_R")
Decorator = Callable[[Callable[_P, _R]], Callable[_P, _R]]


def parametrize(
    arg_name: str,
    test_data: dict[str, Any],
    xfails: dict[str, xfail_type] | None = None,
    skips: dict[str, str] | None = None,
    strict: bool = True,
    flakies: dict[str, int] | None = None,
) -> Decorator:
    """Backend-neutral version of pytest.mark.parametrize with some syntactic
    sugar and added xfail functionality.

    - test_data is expected as a dict of (id, test_data) pairs
    - allows specifying a dict of (id, failure_reason) pairs to mark specific
      tests as xfail. failure_reason can be str or tuple[str, type[Exception]].
      Strings set the reason for failure, the exception type sets the expected
      error.
    """
    xfails = xfails or {}
    skips = skips or {}
    flakies = flakies or {}

    def decorator_func(func: Callable[_P, _R]) -> Callable[_P, _R]:
        pytest_testsuite = []
        for id, test_parameters in test_data.items():
            if id in flakies:
                marker = (pytest.mark.flaky(reruns=flakies[id]),)
            elif id in skips:
                # fail markers do not work with 'buck' based ci, so use skip instead
                marker = (pytest.mark.skip(reason=skips[id]),)
            elif id in xfails:
                xfail_info = xfails[id]
                reason = ""
                raises = None
                if isinstance(xfail_info, str):
                    reason = xfail_info
                elif isinstance(xfail_info, tuple):
                    reason, raises = xfail_info
                else:
                    raise RuntimeError(
                        "xfail info needs to be str, or tuple[str, type[Exception]]"
                    )
                marker = (
                    pytest.mark.xfail(reason=reason, raises=raises, strict=strict),
                )
            else:
                marker = ()

            pytest_param = pytest.param(test_parameters, id=id, marks=marker)
            pytest_testsuite.append(pytest_param)
        decorator = pytest.mark.parametrize(arg_name, pytest_testsuite)
        return decorator(func)

    return decorator_func
