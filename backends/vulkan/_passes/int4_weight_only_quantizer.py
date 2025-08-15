# pyre-unsafe
import logging
from typing import Any, Callable, Dict, Optional, Type

import executorch.backends.vulkan.custom_ops_lib  # noqa

import torch
import torch.nn.functional as F

from torchao.quantization.unified import Quantizer
from torchao.quantization.utils import groupwise_affine_quantize_tensor


# TODO: import from from torchao.quantization.GPTQ.GPTQ import _check_linear_int4_k
# Once diff train catches up
def _check_linear_int4_k(k, group_size=1, inner_k_tiles=None):
    """
    Check if the dimensions are compatible with int4 quantization.

    Args:
        k: The dimension size to check
        group_size: The group size for quantization
        inner_k_tiles: The inner k tiles size

    Returns:
        bool: Whether the dimensions are compatible
    """
    k_divisible_by_group_size = k % group_size == 0
    if inner_k_tiles is not None:
        k_divisible_by_16_times_inner_k_tiles = k % (inner_k_tiles * 16) == 0
        return k_divisible_by_group_size and k_divisible_by_16_times_inner_k_tiles
    return k_divisible_by_group_size


# This module is copied from torchao.quantization.GPTQ.WeightOnlyInt4Linear with
# changes at the annotated lines.
class VkWeightOnlyInt4Linear(torch.nn.Module):
    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        # TODO: remove dtype field, not used
        bias=False,
        device=None,
        dtype=None,
        groupsize: int = 128,
        inner_k_tiles: int = 8,
        precision: torch.dtype = torch.bfloat16,
        scales_precision: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.padding = not _check_linear_int4_k(in_features, groupsize, inner_k_tiles)
        if self.padding:
            from torchao.utils import find_multiple

            self.origin_in_features = in_features
            # pyre-ignore[6]: Incompatible parameter type
            in_features = find_multiple(in_features, 1024)

        self.use_bias = bias
        self.in_features = in_features
        self.out_features = out_features
        self.device = device
        self.groupsize = groupsize
        self.inner_k_tiles = inner_k_tiles
        self.precision = precision
        self.scales_precision = scales_precision

        if dtype is not None:
            raise ValueError("Please specify 'precision' instead of 'dtype'")

        assert out_features % 8 == 0, "require out_features % 8 == 0"
        assert (
            in_features % (inner_k_tiles * 16) == 0
        ), "require in_features % (innerKTiles * 16) == 0"
        # In the original implementation, the weight buffer is registered with the packed
        # sizes, i.e. the result of calling the _convert_weight_to_int4pack operator.
        # However, the Vulkan implementation does not expect the weights to be packed
        # therefore the weight tensor is registered with the unpacked sizes instead.
        # Note that in_features is divided by 2 because each `uint8` tensor element
        # contains 2 4-bit packed values.
        self.register_buffer(
            "weight",
            torch.empty(
                (out_features, in_features // 2),
                dtype=torch.uint8,
                device=device,
            ),
        )
        self.dtype = dtype
        self.register_buffer(
            "scales_and_zeros",
            torch.empty(
                (in_features // groupsize, out_features, 2),
                dtype=self.scales_precision,
                device=device,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.empty((out_features,), dtype=torch.float32, device=device),
            )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.padding:
            input = F.pad(input, pad=(0, self.in_features - self.origin_in_features))
        # The forward method is replaced. In the original implementation, the forward
        # method is torchao.quantization.GPTQ.linear_forward_int4; here a Vulkan custom
        # operator is called instead.
        r = torch.ops.et_vk.linear_weight_int4(
            input,
            self.weight,
            self.groupsize,
            self.scales_and_zeros,
            self.inner_k_tiles,
        )
        if self.use_bias:
            return r + self.bias
        return r


# This function is coped from torchao.quantization.GPTQ._replace_linear_int4
# with small changes at the annotated locations.
def _vk_replace_linear_int4(
    module: torch.nn.Module,
    groupsize: int,
    inner_k_tiles: Optional[int],
    padding_allowed: bool,
    skip_layer_func: Optional[Callable] = None,
    precision: torch.dtype = torch.bfloat16,
    scales_precision: torch.dtype = torch.bfloat16,
    # Use custom vulkan linear layer as default
    linear_class: Type[torch.nn.Module] = VkWeightOnlyInt4Linear,
    copy_weights: bool = False,
):
    for name, child in module.named_children():
        if isinstance(child, torch.nn.Linear) and (
            skip_layer_func is None or not skip_layer_func(child.weight)
        ):
            # Add an additional condition that the out/in features must not exceed the
            # `feature_limit` argument.
            if (
                _check_linear_int4_k(child.in_features, groupsize, inner_k_tiles)
                or padding_allowed
            ):
                new_linear = linear_class(
                    child.in_features,
                    child.out_features,
                    bias=child.bias is not None,
                    device=child.weight.device,
                    groupsize=groupsize,
                    inner_k_tiles=inner_k_tiles,
                    precision=precision,
                    scales_precision=scales_precision,
                )
                if copy_weights and child.weight.device != torch.device("meta"):
                    # pyre-fixme[16]: `Module` has no attribute `weight`.
                    new_linear.weight = child.weight
                    if child.bias is not None:
                        # pyre-fixme[16]: `Module` has no attribute `bias`.
                        new_linear.bias = child.bias
                setattr(module, name, new_linear)
        else:
            _vk_replace_linear_int4(
                child,
                groupsize,
                inner_k_tiles,
                padding_allowed,
                skip_layer_func,
                precision,
                scales_precision,
                linear_class,
                copy_weights,
            )


# This module is copied from torchao.quantization.GPTQ.Int4WeightOnlyQuantizer
# with some changes at the annotated lines.
class VkInt4WeightOnlyQuantizer(Quantizer):
    def __init__(
        self,
        groupsize: int = 256,
        padding_allowed: bool = True,
        inner_k_tiles: Optional[int] = 8,
        device: torch.device = torch.device("cpu"),  # noqa
        precision: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        assert inner_k_tiles in [2, 4, 8]
        assert groupsize in [32, 64, 128, 256]

        self.inner_k_tiles = inner_k_tiles
        self.groupsize: int = groupsize
        self.padding_allowed: bool = padding_allowed
        self.device: torch.device = device
        self.precision: torch.dtype = precision

    @torch.no_grad()
    def _create_quantized_state_dict(
        self, model: torch.nn.Module
    ) -> Dict[str, torch.Tensor]:
        cur_state_dict = model.state_dict()
        for fqn, mod in model.named_modules():
            # Add additional check to make sure features do not exceed feature limit
            if isinstance(mod, torch.nn.Linear):
                out_features = mod.out_features
                in_features = mod.in_features
                logging.info(f"linear: {fqn}, in={in_features}, out={out_features}")

                assert (
                    in_features % self.groupsize == 0
                ), f"require in_features:{in_features} % self.groupsize:{self.groupsize} == 0"

                weight = mod.weight.data
                if not _check_linear_int4_k(
                    in_features, self.groupsize, self.inner_k_tiles
                ):
                    if self.padding_allowed:
                        import torch.nn.functional as F

                        from torchao.utils import find_multiple

                        logging.warn(
                            f"warning: {fqn} is padded to satisfy in_features % 1024 == 0"
                        )
                        # pyre-ignore[6]: Incompatible parameter type
                        padded_in_features = find_multiple(in_features, 1024)
                        weight = F.pad(
                            weight, pad=(0, padded_in_features - in_features)
                        )
                    else:
                        logging.warn(
                            f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            + "and that groupsize and inner_k_tiles*16 evenly divide into it"
                        )
                        continue
                (w_int4x8, scales_and_zeros) = groupwise_affine_quantize_tensor(
                    weight,
                    4,  # n_bit
                    self.groupsize,
                    self.precision,  # dtype for scales_and_zeros
                )
                # If the packing of 2 4-bit values into a single 8-bit value was not
                # performed in the previous function call, then do it manually now.
                if w_int4x8.shape == weight.shape:
                    w_int4x8 = (w_int4x8[::, ::2] << 4 | w_int4x8[::, 1::2]).to(
                        torch.uint8
                    )
                # In the original implementation, w_int4x8 is packed via calling the
                # _convert_weight_to_int4pack operator before storing the weight. However
                # the Vulkan implementation does not expect the weights to be packed, so
                # the w_int4x8 tensor is stored as the weight instead.
                cur_state_dict[f"{fqn}.weight"] = w_int4x8.to(self.device)
                cur_state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros.to(
                    self.device
                )
        return cur_state_dict

    def _convert_for_runtime(self, model: torch.nn.Module) -> torch.nn.Module:
        _vk_replace_linear_int4(
            model,
            self.groupsize,
            self.inner_k_tiles,
            self.padding_allowed,
            skip_layer_func=None,
            precision=self.precision,
            scales_precision=self.precision,
        )
        return model

    def quantize(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:
        state_dict = self._create_quantized_state_dict(model)
        model = self._convert_for_runtime(model)
        model.load_state_dict(state_dict, strict=False)
        return model
