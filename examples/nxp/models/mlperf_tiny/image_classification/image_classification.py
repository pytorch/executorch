# Copyright 2026 NXP
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from tqdm import tqdm

from executorch.examples.nxp.models.mlperf_tiny.mlperf_tiny_model import MLPerfTinyModel
from torchao.quantization.pt2e import disable_observer


INPUT_SHAPE = (1, 3, 32, 32)
NUM_CLASSES = 10

class ImageClassification(MLPerfTinyModel):
    def __init__(self):
        super().__init__()

    @property
    def _input_shape(self):
        return INPUT_SHAPE

    @property
    def _num_classes(self):
        return NUM_CLASSES

    def get_eager_model(self) -> torch.nn.Module:
        return self._model_manager.get_model("image_classification")

    def train_model_fn(
        self, model, num_epochs=15, batch_size=20, channels_last=False
    ):
        torch.manual_seed(42)
        torch.use_deterministic_algorithms(True)

        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=1e-5,
            weight_decay=1e-4,
        )
        loss_fn = torch.nn.CrossEntropyLoss()

        import logging
        logging.warning("Starting training...")

        data = self.get_qat_train_inputs(batch_size=batch_size)
        for nepoch in range(num_epochs):
            for images, labels in tqdm(data):
                if channels_last:
                    images = images.to(memory_format=torch.channels_last)

                optimizer.zero_grad()
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

            if nepoch >= num_epochs / 3:
                model.apply(disable_observer)

        return model
