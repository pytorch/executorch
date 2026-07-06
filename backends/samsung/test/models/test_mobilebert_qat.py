# Copyright (c) Samsung Electronics Co. LTD
# All rights reserved
#
# Licensed under the BSD License (the "License"); you may not use this file
# except in compliance with the License. See the license file in the root
# directory of this source tree for more details.

import argparse
import os
import unittest

import evaluate
import torch

from executorch.backends.samsung.quantizer import EnnQuantizer, Precision
from executorch.backends.samsung.serialization.compile_options import (
    gen_samsung_backend_compile_spec,
)
from executorch.backends.samsung.test.tester import SamsungTester
from executorch.backends.samsung.test.utils.utils import TestConfig
from executorch.examples.samsung.scripts.mobilebert_finetune_QAT import (
    get_dataset,
    MobileBertFinetune,
    trainingQuantModel_QAT,
    transform_attention_mask,
)
from torchao.quantization.pt2e import allow_exported_model_train_eval
from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e


class TestMilestoneMobileBertQAT(unittest.TestCase):
    def test_mobilebert_qat_a8w8(self):
        """
        Test MobileBERT QAT quantization process (A8W8 precision) - using custom QAT training function
        """
        # Create args and metric objects
        args = argparse.Namespace(
            artifact="./mobilebert_qat",
            max_length=256,
            csv_dataset=None,
            batch_size=2,
            num_epochs_for_finetune=1,
        )
        metric = evaluate.load("accuracy")

        os.makedirs(args.artifact, exist_ok=True)

        # Get the finetune model
        mobilebert_finetune = MobileBertFinetune(metric, args)
        model, tokenized_datasets = mobilebert_finetune.get_finetune_mobilebert(
            self.artifact
        )

        # Configure QAT parameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        qat_batch_size = 2  # Batch size for QAT training
        qat_num_epochs = 1  # Number of epochs for QAT training
        qat_training_samples = 10  # In order to quickly test the entire process, only test 10 samples during the qat training
        num_workers = 1  # Number of workers for DataLoader
        calibration_num = 10  # Number of calibration samples

        # Get example inputs
        inputs, _ = get_dataset(
            calibration_num, tokenized_datasets, qat_batch_size, num_workers, device
        )

        example_ref_input_ids = inputs[0][0].to(device)
        example_ref_attention_mask = transform_attention_mask(inputs[0][1].to(device))
        example_inputs = (example_ref_input_ids, example_ref_attention_mask)

        # Export model
        batch_dim = torch.export.Dim("batch_size", min=1, max=qat_batch_size)
        size_input_ids = (qat_batch_size, example_inputs[0].size(1))
        size_attention_mask = (
            qat_batch_size,
            example_inputs[1].size(1),
            example_inputs[1].size(2),
            example_inputs[1].size(3),
        )
        vector_input_ids = torch.randint(0, 256, size_input_ids).to(device)
        vector_attention_mask = torch.zeros(
            size_attention_mask, dtype=torch.float32
        ).to(device)
        export_inputs = (
            vector_input_ids,
            vector_attention_mask,
        )
        exported_model = torch.export.export(
            model.eval().to(device),
            export_inputs,
            dynamic_shapes={
                "input_ids": {0: batch_dim},
                "attention_mask": {0: batch_dim},
            },
        ).module()

        # Prepare QAT model
        quantizer = EnnQuantizer()
        quantizer.setup_quant_params(Precision.A8W8, is_per_channel=True, is_qat=True)
        prepared_model = prepare_pt2e(exported_model, quantizer)

        # Execute QAT training
        qat_dataset = tokenized_datasets.copy()
        qat_dataset["train"] = tokenized_datasets["train"].select(
            range(qat_training_samples)
        )
        trained_model = trainingQuantModel_QAT(
            prepared_model,
            qat_dataset,
            batch_size=qat_batch_size,
            workers=num_workers,
            device=device,
            num_epochs=qat_num_epochs,
        )

        # Convert to quantized model
        quantized_model = convert_pt2e(trained_model)

        # Create SamsungTester and test
        allow_exported_model_train_eval(quantized_model)
        tester = SamsungTester(
            quantized_model,
            example_inputs,
            [gen_samsung_backend_compile_spec(TestConfig.chipset)],
        )

        # Execute test process (skip quantization step, as it has been manually completed)
        (
            tester.export()
            .to_edge_transform_and_lower()
            .to_executorch()
            .run_method_and_compare_outputs(inputs=example_inputs, atol=0.03)
        )
