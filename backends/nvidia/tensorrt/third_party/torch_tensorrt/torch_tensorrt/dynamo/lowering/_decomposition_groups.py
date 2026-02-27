from typing import Any, Callable, Dict, Set, Union

import torch
from torch._decomp import core_aten_decompositions
from torch._decomp import get_decompositions as get_torch_decompositions
from torch._ops import OpOverload, OpOverloadPacket

aten = torch.ops.aten

_core_aten_decompositions: Dict[OpOverload, Callable[[Any], Any]] = (
    core_aten_decompositions()
)
torch_enabled_decompositions: Set[Union[OpOverload, OpOverloadPacket]] = {
    aten._adaptive_avg_pool2d_backward,
    aten.addcdiv,
    aten.addcdiv_,
    aten.addcmul,
    aten.addcmul_,
    aten.addr,
    aten.aminmax,
    aten.arange.default,
    aten.arange.start,
    aten.avg_pool2d_backward,
    aten.binary_cross_entropy,
    aten.binary_cross_entropy_backward,
    aten.binary_cross_entropy_with_logits,
    aten.celu,
    aten.col2im,
    aten.count_nonzero,
    aten.cudnn_batch_norm,
    aten.cudnn_batch_norm_backward,
    aten.deg2rad,
    aten.detach,
    aten.diag_embed,
    aten.diagonal_backward,
    aten.elu_backward,
    aten.embedding_dense_backward,
    aten.empty_like,
    aten._euclidean_dist.default,
    aten.expand_as,
    aten.eye,
    aten.fill,
    aten.frac,
    aten._fused_moving_avg_obs_fq_helper,
    aten.gelu_backward,
    aten.glu_backward,
    aten.hardshrink,
    aten.hardshrink_backward,
    aten.hardsigmoid_backward,
    aten.hardswish,
    aten.hardswish_,
    aten.hardswish_backward,
    aten.hardtanh,
    aten.hardtanh_,
    aten.hardtanh_backward,
    aten.heaviside,
    aten.huber_loss,
    aten.huber_loss_backward,
    aten.im2col,
    aten.index_add,
    aten.index_add_,
    aten.index_copy,
    aten.index_copy_,
    aten.index_fill,
    aten.index_fill_,
    aten.isneginf,
    aten.isposinf,
    aten.l1_loss,
    aten.leaky_relu_,
    aten.leaky_relu_backward,
    aten.lerp,
    aten.linspace,
    aten.logaddexp,
    aten.logaddexp2,
    aten.logit,
    aten.logit_backward,
    aten.log_sigmoid_backward,
    aten.log_sigmoid_forward,
    aten._log_softmax_backward_data,
    aten.logspace,
    aten.logsumexp.default,
    aten.masked_fill,
    aten.masked_fill_,
    aten.max_pool2d_with_indices_backward,
    aten.mish,
    aten.mse_loss,
    aten.mse_loss_backward,
    aten.mvlgamma,
    aten.nansum,
    aten.nan_to_num,
    aten.narrow,
    # TODO: Disable the below operators once freezing is done
    aten.native_batch_norm_backward,
    aten._native_batch_norm_legit_functional,
    aten.native_dropout_backward,
    aten.native_group_norm_backward,
    aten.native_layer_norm_backward,
    aten.new_empty,
    aten.new_full,
    aten.new_ones,
    aten.new_zeros,
    aten.nll_loss_backward,
    aten.nll_loss_forward,
    aten.norm,
    aten.ones,
    aten.ones_like,
    aten._prelu_kernel_backward,
    aten._reshape_alias,
    aten.rad2deg,
    aten.renorm,
    aten.renorm_,
    aten.rot90,
    aten.rsub,
    aten.select_backward,
    aten.select_scatter,
    aten.sgn,
    aten.sigmoid_backward,
    aten.silu_backward,
    aten.sinc,
    aten.slice_backward,
    aten.smooth_l1_loss,
    aten.smooth_l1_loss_backward,
    aten.soft_margin_loss,
    aten.soft_margin_loss_backward,
    aten._softmax.out,
    aten._softmax_backward_data,
    aten.softplus_backward,
    aten.softshrink,
    aten.softshrink_backward,
    aten.special_entr,
    aten.special_log_ndtr,
    aten.special_xlog1py,
    aten.stack,
    aten.std,
    aten.t,
    aten.tanh_backward,
    aten.threshold,
    aten.threshold_backward,
    aten.trace,
    aten.transpose.int,
    aten.tril.default,
    aten.triu.default,
    aten.unbind,
    aten.unfold,
    aten.unfold_backward,
    aten.unfold_copy,
    aten._unsafe_index,
    aten.upsample_nearest2d_backward,
    aten.var,
    aten.var_mean,
    aten.xlogy,
    aten.zero,
    aten.zero_,
    aten.zeros,
    aten.zeros_like,
    # Non-default convenience decompositions
    aten.clamp_min,
    aten.clamp_max,
    aten.linalg_vector_norm,
    aten.repeat,
}
torch_disabled_decompositions: Set[Union[OpOverload, OpOverloadPacket]] = {
    aten._softmax.default,
    aten.upsample_nearest1d.vec,
    aten.upsample_nearest2d.vec,
    aten.upsample_nearest3d.vec,
    aten.upsample_linear1d.vec,
    aten.upsample_bilinear2d.vec,
    aten.upsample_trilinear3d.vec,
    aten.upsample_bicubic2d.vec,
    aten.linear.default,
    aten.matmul.default,
}


ENABLED_TORCH_DECOMPOSITIONS: Dict[OpOverload, Callable[[Any], Any]] = (
    get_torch_decompositions(torch_enabled_decompositions)
)
TORCH_TRT_DECOMPOSITIONS: Dict[OpOverload, Callable[[Any], Any]] = {}


def check_decomp_set_invariants() -> None:
    """Validates no overlap between enabled and disabled decomposition sets"""
    overlap = torch_enabled_decompositions.intersection(torch_disabled_decompositions)

    if overlap:
        raise AssertionError(
            f"Detected {overlap} registered in both torch_enabled_decompositions "
            "and torch_disabled_decompositions. Ensure all operator(s) are in "
            "at most one of the two sets."
        )


check_decomp_set_invariants()
