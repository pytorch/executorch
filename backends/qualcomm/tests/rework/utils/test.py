# Copyright (c) Qualcomm Innovation Center, Inc.
# All rights reserved
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from executorch.backends.qualcomm.tests.rework.src.utils import *  # noqa: F403


def test_dump_context_from_pte(quantizer, compile_spec):
    DumpContextFromPte.test(quantizer, compile_spec)  # noqa: F405


def test_draw_graph(quantizer):
    DrawGraph.test(quantizer)  # noqa: F405


def test_fixed_point_floating_point_mixed_precision(subtests, quantizer, compile_spec):
    FixedPointFloatingPointMixedPrecision.test(  # noqa: F405
        subtests, quantizer, compile_spec
    )


def test_multi_contexts_composite(compile_spec):
    MultiContextsComposite.test(compile_spec)  # noqa: F405


def test_rewrite_prepared_observer(quantizer):
    RewritePreparedObserver.test(quantizer)  # noqa: F405


def test_skip_node_partitioner(subtests, quantizer, compile_spec):
    SkipNodePartitioner.test(subtests, quantizer, compile_spec)  # noqa: F405


def test_skip_node_quantizer(subtests, quantizer, compile_spec):
    SkipNodeQuantizer.test(subtests, quantizer, compile_spec)  # noqa: F405


def test_qat(subtests):
    QAT.test(subtests)  # noqa: F405
