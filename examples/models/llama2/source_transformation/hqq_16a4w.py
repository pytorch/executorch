# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from executorch.examples.models.llama2.evaluate import EagerEvalWrapper, evaluate_model
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear

########################## Run HQQ ###############################


def _replace_linear_4w_hqq(
    module: torch.nn.Module,
    quant_config,
    compute_dtype,
    del_orig=False,
):
    """
    Recursively replacing all Linear layers with HQQLinear with the 4bit quantized weights
    """
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear):
            new_linear = HQQLinear(
                child,
                quant_config,
                compute_dtype=compute_dtype,
                del_orig=True,
                device="cpu",
            )
            setattr(module, name, new_linear)
        else:
            _replace_linear_4w_hqq(
                child,
                quant_config,
                compute_dtype,
                del_orig=False,
            )


def replace_linear_4w_hqq(
    module: torch.nn.Module,
    quant_config: BaseQuantizeConfig,
    compute_dtype,
    del_orig=False,
):
    """
    Replace all Linear layers with HQQLinear with the 4bit quantized weights
    """
    _replace_linear_4w_hqq(
        module,
        quant_config,
        compute_dtype,
        del_orig=False,
    )


def run_hqq_quantize(model: torch.nn.Module) -> None:
    """
    Inplace update the model with the hqq quantized weights
    """

    quant_config = BaseQuantizeConfig(
        quant_zero=False, quant_scale=False, offload_meta=False, view_as_float=False
    )

    replace_linear_4w_hqq(model, quant_config=quant_config, compute_dtype=torch.float32)


########################## Use static quantization with HQQ Linear ###############################


def calibrate(
    model, tokenizer, calibration_tasks, calibration_limit, calibration_seq_length
):
    print("run calibration...")
    eval_wrapper = EagerEvalWrapper(
        model=model,
        tokenizer=tokenizer,
        max_seq_length=calibration_seq_length,
        use_kv_cache=False,
    )
    eval_results = evaluate_model(
        eval_wrapper,
        tasks=calibration_tasks,
        limit=calibration_limit,
    )
    for task, res in eval_results["results"].items():
        print(f"Reference result with hqq model: {task}: {res}")


class LinearActivationFakeQuant(torch.nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.linear = linear
        self.input_activation_fake_quant = torch.quantization.FakeQuantize(
            observer=torch.quantization.MovingAverageMinMaxObserver,
            dtype=torch.int32,
            quant_min=torch.iinfo(torch.uint16).min,
            quant_max=torch.iinfo(torch.uint16).max,
        )
        self.output_activation_fake_quant = torch.quantization.FakeQuantize(
            observer=torch.quantization.MovingAverageMinMaxObserver,
            dtype=torch.int32,
            quant_min=torch.iinfo(torch.uint16).min,
            quant_max=torch.iinfo(torch.uint16).max,
        )

    def forward(self, x):
        x = self.input_activation_fake_quant(x)
        return self.output_activation_fake_quant(self.linear(x))


def get_quant_params(activation_fake_quant):
    quant_min = activation_fake_quant.quant_min
    quant_max = activation_fake_quant.quant_max
    qparams = activation_fake_quant.calculate_qparams()
    scale = qparams[0]
    zero_point = qparams[1]
    return (quant_min, quant_max, scale, zero_point)


class LinearActivationQuant(torch.nn.Module):

    def __init__(self, linear_fake_quant):
        super().__init__()
        self.linear_fake_quant = linear_fake_quant
        (
            self.input_quant_min,
            self.input_quant_max,
            self.input_scale,
            self.input_zero_point,
        ) = get_quant_params(linear_fake_quant.input_activation_fake_quant)

        (
            self.output_quant_min,
            self.output_quant_max,
            self.output_scale,
            self.output_zero_point,
        ) = get_quant_params(linear_fake_quant.output_activation_fake_quant)

    def forward(self, x):
        # Manually quantize the input tensor using observed min and max values
        q_tensor = torch.round(x / self.input_scale + self.input_zero_point)
        # Clip to ensure within the range [quant min and quant max]
        q_tensor = torch.clamp(q_tensor, self.input_quant_min, self.input_quant_max)
        # Dequantize to the original scale
        dequantized_tensor = (q_tensor - self.input_zero_point) * self.input_scale

        linear_output = self.linear_fake_quant.linear(dequantized_tensor)

        # # Quantize the linear output tensor
        q_linear_output = torch.round(
            linear_output / self.output_scale + self.output_zero_point
        )
        q_linear_output = torch.clamp(
            q_linear_output, self.output_quant_min, self.output_quant_max
        )
        # Dequantize the linear output tensor
        dq_linear_output = (
            q_linear_output - self.output_zero_point
        ) * self.output_scale

        return dq_linear_output


def _replace_linear_quant_activation(module: torch.nn.Module, stage: str):
    for name, child in module.named_children():
        if stage == "convert":
            if isinstance(child, LinearActivationFakeQuant):
                new_linear = LinearActivationQuant(child)
                setattr(module, name, new_linear)
            else:
                _replace_linear_quant_activation(child, stage)
        elif stage == "prepare":
            if isinstance(child, HQQLinear):
                new_linear = LinearActivationFakeQuant(child)
                setattr(module, name, new_linear)
            else:
                _replace_linear_quant_activation(child, stage)
        else:
            raise ValueError(f"Unsupported stage {stage}")


def replace_linear_quant_activation(module: torch.nn.Module, stage: str):
    _replace_linear_quant_activation(
        module,
        stage,
    )


def prepare(model):
    """
    Prepare the model for quantization by manually inserting the observors
    """
    replace_linear_quant_activation(model, "prepare")


def convert(model):
    """
    Convert the observors the actual quant/dequant nodes, in this implementation, we manually
    calling add, mul, clamp for quick prototyping
    """
    replace_linear_quant_activation(model, "convert")
