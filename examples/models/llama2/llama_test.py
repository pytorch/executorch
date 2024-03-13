# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from .export_llama import build_model


class LlamaTest(unittest.TestCase):
    def test_quantized_llama(self):
        _ = build_model(
            modelname="model",
            extra_opts="--fairseq2 -qmode int8",
            par_local_output=True,
            resource_pkg_name=__name__,
        )

    def test_half_llama(self):
        _ = build_model(
            modelname="model",
            extra_opts="--fairseq2 -d fp16",
            par_local_output=True,
            resource_pkg_name=__name__,
        )


#    def test_half_xnnpack_llama(self):
#        output_path = build_model(
#            modelname="model",
#            extra_opts="--fairseq2 -d fp16 -X",
#            par_local_output=True,
#            resource_pkg_name=__name__,
#        )
