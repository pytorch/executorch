# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from timm.models import inception_v4

FORMAT = "[%(filename)s:%(lineno)s] %(message)s"
logging.basicConfig(format=FORMAT)


# will refactor this in a separate file.
class InceptionV4Model:
    def __init__(self):
        pass

    @staticmethod
    def get_model():
        logging.info("loading inception_v4 model")
        m = inception_v4()
        logging.info("loaded inception_v4 model")
        return m

    @staticmethod
    def get_example_inputs():
        return (torch.randn(3, 299, 299).unsqueeze(0),)
