# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch

from ultralytics import YOLO

from ..model_base import EagerModelBase


class YOLO26Model(EagerModelBase):
    def __init__(self, model_name: str = "yolo26n.pt"):
        self.model_name = model_name
        self.yolo = YOLO(model_name)
        self.dummy_frame = torch.randn((320, 320, 3)).to(torch.uint8).numpy()
        # nms=False selects the NMS-free head, whose [1, max_det, 6] output the
        # runner expects; fuse() drops the other branch.
        self.yolo.predict(self.dummy_frame, imgsz=(320, 320), verbose=False, nms=False)
        # Since ultralytics 8.4.140 predict() runs on a copy of the model. Export
        # that fused, warmed-up copy: a cold head rebuilds its anchors while being
        # traced, and the unbacked float this creates fails strict export.
        # predict() also converts the copy to channels_last on x86 CPUs; undo it.
        self.model = self.yolo.predictor.model.model.eval().to(
            memory_format=torch.contiguous_format
        )

    def get_eager_model(self) -> torch.nn.Module:
        logging.info("Loading " + self.model_name + " model")
        m = self.model
        logging.info("Loaded " + self.model_name + " model")
        return m

    def get_example_inputs(self):
        return (self.yolo.predictor.preprocess([self.dummy_frame]),)
