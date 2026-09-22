# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager

SHORT_QUERY_MAX_ROWS = 4


@contextmanager
def short_query_kernels(max_rows: int = 16):
    """Opt into wider packed kernels while tracing a short-query method.

    The default stays at four rows so existing dynamic prefill exports do not
    acquire a new shape boundary. This process-wide scope requires serial export.
    """
    if not 4 <= max_rows <= 16:
        raise ValueError("max_rows must be in [4, 16]")
    global SHORT_QUERY_MAX_ROWS
    previous = SHORT_QUERY_MAX_ROWS
    SHORT_QUERY_MAX_ROWS = max_rows
    try:
        yield
    finally:
        SHORT_QUERY_MAX_ROWS = previous
