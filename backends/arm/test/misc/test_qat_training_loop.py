# Copyright 2025 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging

import torch
from executorch.backends.arm.quantizer import (
    get_symmetric_quantization_config,
    TOSAQuantizer,
)

from executorch.backends.arm.tosa.specification import TosaSpecification
from torch.export import export
from torchao.quantization.pt2e import (
    move_exported_model_to_eval,
    move_exported_model_to_train,
)
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_qat_pt2e

logger = logging.getLogger(__name__)


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sequential = torch.nn.Sequential(
            torch.nn.Linear(1, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 1),
        )

    def forward(self, x):
        return self.sequential(x)


def evaluate_model(model, inputs, expected_outputs):
    with torch.no_grad():
        test_outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(test_outputs, expected_outputs)
        logger.info(f"Mean squared error: {loss.item()}")


def test_qat_training_loop_tosa_INT():
    """Test the QAT training loop with a simple MLP model.
    This function creates a simple MLP model, prepares it for QAT, runs a training loop,
    and evaluates the quantized model to make sure everything works as expected."""

    model = MLP()
    logger.info("Starting training loop test")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(100):
        model.train()
        optimizer.zero_grad()
        inputs = torch.randn(100, 1).clamp(-1, 1)
        outputs = model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, torch.sin(inputs))
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            logger.info(f"Epoch {epoch}, Loss: {loss.item()}")
    logger.info("Training loop test completed successfully")

    logger.info("Evaluating model before QAT")
    test_inputs = torch.randn(20, 1).clamp(-1, 1)
    test_outputs = torch.sin(test_inputs)
    evaluate_model(model, test_inputs, test_outputs)

    exported_model = export(model, (torch.randn(1, 1),), strict=True)
    quantizer = TOSAQuantizer(TosaSpecification.create_from_string("TOSA-1.0+INT"))
    quantizer.set_global(get_symmetric_quantization_config(is_qat=True))

    prepared_model = prepare_qat_pt2e(exported_model.module(), quantizer)
    prepared_model = move_exported_model_to_train(prepared_model)
    logger.info("QAT model prepared successfully")

    logger.info("Starting QAT training loop")

    for epoch in range(25):
        inputs = torch.randn(100, 1).clamp(-1, 1)
        optimizer.zero_grad()
        outputs = prepared_model(inputs)
        loss = torch.nn.functional.mse_loss(outputs, torch.sin(inputs))
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            logger.info(f"QAT Epoch {epoch}, Loss: {loss.item()}")
    logger.info("QAT training loop completed successfully")
    prepared_model = move_exported_model_to_eval(prepared_model)

    quantized_model = convert_pt2e(prepared_model)
    logger.info("QAT model quantized successfully")

    logger.info("Evaluating quantized model")
    test_inputs = torch.randn(100, 1).clamp(-1, 1)
    test_outputs = torch.sin(test_inputs)
    evaluate_model(quantized_model, test_inputs, test_outputs)
