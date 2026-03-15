from typing import Any, Optional, Tuple

import numpy as np
import tensorrt as trt
import torch
from tensorrt import ITensor as TRTTensor
from torch.fx.node import Target
from torch_tensorrt.dynamo._SourceIR import SourceIR
from torch_tensorrt.dynamo.conversion._ConversionContext import ConversionContext
from torch_tensorrt.dynamo.conversion.impl.activation.base import convert_activation


def relu(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    operation_type = trt.ActivationType.RELU

    def relu_dyn_range_fn(dyn_range: Tuple[float, float]) -> Tuple[float, float]:
        return max(0, dyn_range[0]), max(0, dyn_range[1])

    return convert_activation(
        ctx,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        dyn_range_fn=relu_dyn_range_fn,
    )


def sigmoid(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    operation_type = trt.ActivationType.SIGMOID

    def sigmoid_dyn_range_fn(dyn_range: Tuple[float, float]) -> Tuple[float, float]:
        def sigmoid_fn(x: float) -> Any:
            return 1 / (1 + np.exp(-x))

        return sigmoid_fn(dyn_range[0]), sigmoid_fn(dyn_range[1])

    return convert_activation(
        ctx,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        dyn_range_fn=sigmoid_dyn_range_fn,
    )


def tanh(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    operation_type = trt.ActivationType.TANH

    def tanh_dyn_range_fn(dyn_range: Tuple[float, float]) -> Tuple[float, float]:
        def tanh_fn(x: float) -> Any:
            return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

        return tanh_fn(dyn_range[0]), tanh_fn(dyn_range[1])

    return convert_activation(
        ctx,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        dyn_range_fn=tanh_dyn_range_fn,
    )


def leaky_relu(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    alpha: float = 0.01,
) -> TRTTensor:
    operation_type = trt.ActivationType.LEAKY_RELU

    def leaky_relu_dyn_range_fn(dyn_range: Tuple[float, float]) -> Tuple[float, float]:
        def leaky_relu_fn(x: float) -> float:
            return max(0, x) + alpha * min(0, x)

        return leaky_relu_fn(dyn_range[0]), leaky_relu_fn(dyn_range[1])

    return convert_activation(
        ctx,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        alpha,
        dyn_range_fn=leaky_relu_dyn_range_fn,
    )


def elu(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    alpha: float = 1.0,
    beta: Optional[float] = None,
) -> TRTTensor:
    EPS = 1e-4
    # actually call selu()
    if (
        abs(alpha - 1.6732632423543772) < EPS
        and beta is not None
        and abs(beta - 1.0507009873554805) < EPS
    ):
        print("Selu is called but re-uses elu function!")
        return selu(ctx, target, source_ir, name, input_val)

    else:
        operation_type = trt.ActivationType.ELU

        def elu_dyn_range_fn(dyn_range: Tuple[float, float]) -> Tuple[float, float]:
            return (
                torch.nn.functional.elu(dyn_range[0], alpha),
                torch.nn.functional.elu(dyn_range[1], alpha),
            )

        return convert_activation(
            ctx,
            target,
            source_ir,
            name,
            operation_type,
            input_val,
            alpha,
            dyn_range_fn=elu_dyn_range_fn,
        )


def selu(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    operation_type = trt.ActivationType.SELU

    def selu_dyn_range_fn(dyn_range: Tuple[float, float]) -> Tuple[float, float]:
        return (
            torch.nn.functional.selu(dyn_range[0]),
            torch.nn.functional.selu(dyn_range[1]),
        )

    return convert_activation(
        ctx,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        dyn_range_fn=selu_dyn_range_fn,
    )


# no corresponding function in aten/native_functions
def softsign(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
) -> TRTTensor:
    operation_type = trt.ActivationType.SOFTSIGN

    def softsign_dyn_range_fn(dyn_range: Tuple[float, float]) -> Tuple[float, float]:
        return (
            torch.nn.functional.softsign(dyn_range[0]),
            torch.nn.functional.softsign(dyn_range[1]),
        )

    return convert_activation(
        ctx,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        dyn_range_fn=softsign_dyn_range_fn,
    )


def softplus(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    beta: float = 1,
) -> TRTTensor:
    operation_type = trt.ActivationType.SOFTPLUS

    def softplus_dyn_range_fn(dyn_range: Tuple[float, float]) -> Tuple[float, float]:
        return (
            torch.nn.functional.softplus(dyn_range[0], beta),
            torch.nn.functional.softplus(dyn_range[1], beta),
        )

    return convert_activation(
        ctx,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        alpha=1 / beta,
        beta=beta,
        dyn_range_fn=softplus_dyn_range_fn,
    )


def hard_sigmoid(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    alpha: float,
    beta: float,
) -> TRTTensor:
    operation_type = trt.ActivationType.HARD_SIGMOID

    def hard_sigmoid_dyn_range_fn(
        dyn_range: Tuple[float, float],
    ) -> Tuple[float, float]:
        def hard_sigmoid_fn(x: float) -> float:
            return max(0, min(1, alpha * x + beta))

        return hard_sigmoid_fn(dyn_range[0]), hard_sigmoid_fn(dyn_range[1])

    return convert_activation(
        ctx,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        alpha=alpha,
        beta=beta,
        dyn_range_fn=hard_sigmoid_dyn_range_fn,
    )


# no corresponding function in aten/native_functions
def scaled_tanh(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    alpha: float,
    beta: float,
) -> TRTTensor:
    operation_type = trt.ActivationType.SCALED_TANH

    def scaled_tanh_dyn_range_fn(dyn_range: Tuple[float, float]) -> Tuple[float, float]:
        def scaled_tanh_fn(x: float) -> Any:
            return alpha * torch.nn.functional.tanh(beta * x)

        return scaled_tanh_fn(dyn_range[0]), scaled_tanh_fn(dyn_range[1])

    return convert_activation(
        ctx,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        alpha=alpha,
        beta=beta,
        dyn_range_fn=scaled_tanh_dyn_range_fn,
    )


# no corresponding function in aten/native_functions
def thresholded_relu(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    alpha: float,
) -> TRTTensor:
    operation_type = trt.ActivationType.THRESHOLDED_RELU

    def thresholded_relu_dyn_range_fn(
        dyn_range: Tuple[float, float],
    ) -> Tuple[float, float]:
        def thresholded_relu_fn(x: float) -> float:
            return x if x > alpha else 0

        return thresholded_relu_fn(dyn_range[0]), thresholded_relu_fn(dyn_range[1])

    return convert_activation(
        ctx,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
        alpha=alpha,
        dyn_range_fn=thresholded_relu_dyn_range_fn,
    )


def gelu(
    ctx: ConversionContext,
    target: Target,
    source_ir: Optional[SourceIR],
    name: str,
    input_val: TRTTensor,
    approximate: str,
) -> TRTTensor:
    if approximate == "none":
        operation_type = trt.ActivationType.GELU_ERF
    elif approximate == "tanh":
        operation_type = trt.ActivationType.GELU_TANH

    return convert_activation(
        ctx,
        target,
        source_ir,
        name,
        operation_type,
        input_val,
    )
