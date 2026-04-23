# Copyright 2026 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Set, Type

from executorch.backends.arm._passes import ArmPass
from executorch.backends.arm._passes.convert_full_like_to_full_pass import (
    ConvertFullLikeToFullPass,
)
from executorch.backends.arm._passes.decompose_sqrt_pass import DecomposeSqrtPass
from executorch.backends.arm._passes.match_arg_dtype_pass import MatchArgDtypePass
from executorch.backends.arm._passes.match_arg_ranks_pass import MatchArgRanksPass
from executorch.backends.arm._passes.replace_scalar_with_tensor_pass import (
    ReplaceScalarWithTensorByProfilePass,
)
from executorch.exir.dialects._ops import ops as exir_ops
from executorch.exir.pass_base import ExportPass


edge_erfinv_ops = (exir_ops.edge.aten.erfinv.default,)


def get_erfinv_decomposition(op) -> tuple:
    if op in edge_erfinv_ops:
        # Ordered by first use in call_operator below.
        return (
            exir_ops.edge.aten.lt.Tensor,
            exir_ops.edge.aten.where.self,
            exir_ops.edge.aten.abs.default,
            exir_ops.edge.aten.gt.Tensor,
            exir_ops.edge.aten.eq.Tensor,
            exir_ops.edge.aten.mul.Tensor,
            exir_ops.edge.aten.mul.Scalar,
            exir_ops.edge.aten.add.Scalar,
            exir_ops.edge.aten.add.Tensor,
            exir_ops.edge.aten.div.Tensor,
            exir_ops.edge.aten.sub.Tensor,
            exir_ops.edge.aten.log.default,
            exir_ops.edge.aten.sqrt.default,
            exir_ops.edge.aten.lt.Scalar,
            exir_ops.edge.aten.erf.default,
            exir_ops.edge.aten.exp.default,
            exir_ops.edge.aten.gt.Scalar,
        )

    raise RuntimeError(f"Can't get erfinv decomposition for op {op}")


class DecomposeErfinvPass(ArmPass):
    """Decomposes `aten.erfinv` using the same *initial-guess* approximation as
    the PyTorch CPU scalar `calc_erfinv`, with a guarded Newton refinement step
    to improve numerical accuracy (especially for fp16).

    Source:
      - PyTorch `calc_erfinv` (CPU scalar fallback), Math.h (pinned commit):
        https://github.com/pytorch/pytorch/blob/f61180da7043f27a1d8c5aec88fd4b910aca987a/aten/src/ATen/native/Math.h#L152

    The approximation is piecewise:

    Domain / special cases
      - For |x| > 1:  erfinv(x) = NaN
      - For  x =  1:  erfinv(x) = +inf
      - For  x = -1:  erfinv(x) = -inf

    Definitions
      - s = sign(x)  (s = -1 if x < 0 else +1)
      - a = |x|

    Branch selection
      - Central branch if a < 0.7
      - Tail branch otherwise (a >= 0.7)

    ------------------------------------------------------------------------
    Central branch (a < 0.7)
    ------------------------------------------------------------------------
      Let z = x^2. Define:

        P(z) = a0 + a1*z + a2*z^2 + a3*z^3
        Q(z) = 1  + b0*z + b1*z^2 + b2*z^3 + b3*z^4

      Initial guess:

        y0 = x * P(z) / Q(z)

      Coefficients:
        a0= 0.886226899   a1=-1.645349621   a2= 0.914624893   a3=-0.140543331
        b0=-2.118377725   b1= 1.442710462   b2=-0.329097515   b3= 0.012229801

    ------------------------------------------------------------------------
    Tail branch (a >= 0.7)
    ------------------------------------------------------------------------
      Compute:

        u  = 0.5 * (1 - a)
        t  = sqrt( -log(u) )

      Note: Computing `log(u)` directly often gives better numerical accuracy than
      forming an equivalent expression using `log1p(-a)` near a -> 1 (where 1-a is
      tiny), because the transformation through `log1p` can introduce larger relative
      error in that regime.

      Define:

        R(t) = c0 + c1*t + c2*t^2 + c3*t^3
        S(t) = 1  + d0*t + d1*t^2

      Initial guess:

        y0 = s * | R(t) / S(t) |

      Coefficients:
        c0=-1.970840454   c1=-1.624906493   c2= 3.429567803   c3= 1.641345311
        d0= 3.543889200   d1= 1.637067800

    Output
      Returns the refined estimate (starting from y0) with the special-case handling
      for |x|>1 and x=±1 as described above.

    """

    _passes_required_after: Set[Type[ExportPass]] = {
        DecomposeSqrtPass,
        ConvertFullLikeToFullPass,
        MatchArgRanksPass,
        MatchArgDtypePass,
        ReplaceScalarWithTensorByProfilePass,
    }

    def call_operator(self, op, args, kwargs, meta):
        if op not in edge_erfinv_ops:
            return super().call_operator(op, args, kwargs, meta, updated=False)

        if self._is_quantized_meta(meta):
            # If quantized, node should be replaced by table op.
            return super().call_operator(op, args, kwargs, meta)

        x = args[0]

        (
            op_lt_t,
            op_where,
            op_abs,
            op_gt_t,
            op_eq_t,
            op_mul_t,
            op_mul_s,
            op_add_s,
            op_add_t,
            op_div_t,
            op_sub_t,
            op_log,
            op_sqrt,
            op_lt_s,
            op_erf,
            op_exp,
            op_gt_s,
        ) = get_erfinv_decomposition(op)

        # ---- constants (PyTorch calc_erfinv coefficients) ----
        CENTRAL_RANGE = 0.7

        # central: x * P(z)/Q(z), z = x^2
        a0, a1, a2, a3 = 0.886226899, -1.645349621, 0.914624893, -0.140543331
        b0, b1, b2, b3 = -2.118377725, 1.442710462, -0.329097515, 0.012229801

        # tail: s * |R(t)/S(t)|, t = sqrt(-log((1-|x|)/2))
        c0, c1, c2, c3 = -1.970840454, -1.624906493, 3.429567803, 1.641345311
        d0, d1 = 3.543889200, 1.637067800

        # Newton refinement controls (safe defaults for fp16)
        #   y1 = y0 - (erf(y0) - x) / ((2/sqrt(pi)) * exp(-y0^2))
        NEWTON_ITERS = 1
        REFINE_MAX_ABS = 0.95
        DERIV_EPS = 1e-3
        CORR_MAX = 0.5
        TWO_OVER_SQRT_PI = 1.1283791670955126

        # ---- zeros / ones constants ----
        zeros = super().call_scalar(0.0, meta)
        ones = super().call_scalar(1.0, meta)
        neg_ones = super().call_scalar(-1.0, meta)

        # ---- s = sign(x): -1 for x<0 else +1 ----
        x_lt0 = super().call_operator(op_lt_t, (x, zeros), {}, meta, updated=True)
        s = super().call_operator(
            op_where, (x_lt0, neg_ones, ones), {}, meta, updated=True
        )

        # ---- a = |x| and domain masks ----
        a = super().call_operator(op_abs, (x,), {}, meta, updated=True)
        abs_gt1 = super().call_operator(
            op_gt_t, (a, ones), {}, meta, updated=True
        )  # |x| > 1
        abs_eq1 = super().call_operator(
            op_eq_t, (a, ones), {}, meta, updated=True
        )  # |x| == 1

        # For internal math, avoid evaluating log(0), sqrt(inf), etc. on |x|==1 or |x|>1 lanes.
        a_math = super().call_operator(
            op_where, (abs_gt1, zeros, a), {}, meta, updated=True
        )
        a_math = super().call_operator(
            op_where, (abs_eq1, zeros, a_math), {}, meta, updated=True
        )

        # ------------------------------------------------------------------
        # Central initial guess: y0 = x * P(z) / Q(z), z = x^2
        # ------------------------------------------------------------------
        z = super().call_operator(op_mul_t, (x, x), {}, meta, updated=True)  # z = x^2

        # P(z) via Horner: (((a3*z + a2)*z + a1)*z + a0)
        P = super().call_operator(op_mul_s, (z, a3), {}, meta, updated=True)
        P = super().call_operator(op_add_s, (P, a2), {}, meta, updated=True)
        P = super().call_operator(op_mul_t, (P, z), {}, meta, updated=True)
        P = super().call_operator(op_add_s, (P, a1), {}, meta, updated=True)
        P = super().call_operator(op_mul_t, (P, z), {}, meta, updated=True)
        P = super().call_operator(op_add_s, (P, a0), {}, meta, updated=True)

        # Q(z) via Horner: ((((b3*z + b2)*z + b1)*z + b0)*z + 1)
        Q = super().call_operator(op_mul_s, (z, b3), {}, meta, updated=True)
        Q = super().call_operator(op_add_s, (Q, b2), {}, meta, updated=True)
        Q = super().call_operator(op_mul_t, (Q, z), {}, meta, updated=True)
        Q = super().call_operator(op_add_s, (Q, b1), {}, meta, updated=True)
        Q = super().call_operator(op_mul_t, (Q, z), {}, meta, updated=True)
        Q = super().call_operator(op_add_s, (Q, b0), {}, meta, updated=True)
        Q = super().call_operator(op_mul_t, (Q, z), {}, meta, updated=True)
        Q = super().call_operator(op_add_t, (Q, ones), {}, meta, updated=True)

        xP = super().call_operator(op_mul_t, (x, P), {}, meta, updated=True)
        y0_central = super().call_operator(op_div_t, (xP, Q), {}, meta, updated=True)

        # ------------------------------------------------------------------
        # Tail initial guess: y0 = s * |R(t)/S(t)|
        #   u = 0.5*(1-a), t = sqrt(-log(u))
        # ------------------------------------------------------------------
        one_minus_a = super().call_operator(
            op_sub_t, (ones, a_math), {}, meta, updated=True
        )  # 1-a
        u = super().call_operator(
            op_mul_s, (one_minus_a, 0.5), {}, meta, updated=True
        )  # u = 0.5*(1-a)

        # Avoid log(0) poisoning intermediates: for a==1 lanes, feed log(1)=0 and fix later.
        u_safe = super().call_operator(
            op_where, (abs_eq1, ones, u), {}, meta, updated=True
        )
        log_u = super().call_operator(op_log, (u_safe,), {}, meta, updated=True)
        neg_log_u = super().call_operator(
            op_mul_t, (log_u, neg_ones), {}, meta, updated=True
        )
        t = super().call_operator(
            op_sqrt, (neg_log_u,), {}, meta, updated=True
        )  # t = sqrt(-log(u))

        # R(t) via Horner: ((c3*t + c2)*t + c1)*t + c0
        R = super().call_operator(op_mul_s, (t, c3), {}, meta, updated=True)
        R = super().call_operator(op_add_s, (R, c2), {}, meta, updated=True)
        R = super().call_operator(op_mul_t, (R, t), {}, meta, updated=True)
        R = super().call_operator(op_add_s, (R, c1), {}, meta, updated=True)
        R = super().call_operator(op_mul_t, (R, t), {}, meta, updated=True)
        R = super().call_operator(op_add_s, (R, c0), {}, meta, updated=True)

        # S(t) via Horner: (d1*t + d0)*t + 1
        S = super().call_operator(op_mul_s, (t, d1), {}, meta, updated=True)
        S = super().call_operator(op_add_s, (S, d0), {}, meta, updated=True)
        S = super().call_operator(op_mul_t, (S, t), {}, meta, updated=True)
        S = super().call_operator(op_add_t, (S, ones), {}, meta, updated=True)

        frac = super().call_operator(op_div_t, (R, S), {}, meta, updated=True)
        frac_abs = super().call_operator(op_abs, (frac,), {}, meta, updated=True)
        y0_tail = super().call_operator(op_mul_t, (s, frac_abs), {}, meta, updated=True)

        # ---- select central vs tail (use lt to avoid le-lowering quirks) ----
        in_central = super().call_operator(
            op_lt_s, (a, CENTRAL_RANGE), {}, meta, updated=True
        )
        y0 = super().call_operator(
            op_where, (in_central, y0_central, y0_tail), {}, meta, updated=True
        )

        # Ensure y0 doesn’t carry inf/nan before refinement / final where.
        y0 = super().call_operator(
            op_where, (abs_gt1, zeros, y0), {}, meta, updated=True
        )
        y0 = super().call_operator(
            op_where, (abs_eq1, zeros, y0), {}, meta, updated=True
        )

        # ------------------------------------------------------------------
        # Guarded Newton refinement
        #
        # We want to solve erf(y) = x for y.
        #
        # Newton's method update:
        #   y_{k+1} = y_k - f(y_k) / f'(y_k)
        # where:
        #   f(y)  = erf(y) - x
        #   f'(y) = d/dy erf(y) = (2/sqrt(pi)) * exp(-y^2)
        #
        # So the refinement step is:
        #   y_{k+1} = y_k - (erf(y_k) - x) / ((2/sqrt(pi)) * exp(-y_k^2))
        #
        # Guards:
        #   - only apply refinement for |x| < REFINE_MAX_ABS (avoid tail instability)
        #   - skip the update if f'(y_k) is tiny (DERIV_EPS) to avoid huge steps
        #   - skip the update if |correction| is too large (CORR_MAX) to avoid overshoot
        # ------------------------------------------------------------------
        refine_mask = super().call_operator(
            op_lt_s, (a, REFINE_MAX_ABS), {}, meta, updated=True
        )

        y = y0
        for _ in range(NEWTON_ITERS):
            erf_y = super().call_operator(op_erf, (y,), {}, meta, updated=True)
            err = super().call_operator(op_sub_t, (erf_y, x), {}, meta, updated=True)

            y_sq = super().call_operator(op_mul_t, (y, y), {}, meta, updated=True)
            neg_y_sq = super().call_operator(
                op_mul_t, (y_sq, neg_ones), {}, meta, updated=True
            )
            exp_term = super().call_operator(
                op_exp, (neg_y_sq,), {}, meta, updated=True
            )

            deriv = super().call_operator(
                op_mul_s, (exp_term, TWO_OVER_SQRT_PI), {}, meta, updated=True
            )

            deriv_tiny = super().call_operator(
                op_lt_s, (deriv, DERIV_EPS), {}, meta, updated=True
            )
            corr = super().call_operator(op_div_t, (err, deriv), {}, meta, updated=True)
            corr_abs = super().call_operator(op_abs, (corr,), {}, meta, updated=True)
            corr_huge = super().call_operator(
                op_gt_s, (corr_abs, CORR_MAX), {}, meta, updated=True
            )

            y1 = super().call_operator(op_sub_t, (y, corr), {}, meta, updated=True)

            # Apply guards: if tiny deriv or huge correction -> keep y
            y_safe = super().call_operator(
                op_where, (deriv_tiny, y, y1), {}, meta, updated=True
            )
            y_safe = super().call_operator(
                op_where, (corr_huge, y, y_safe), {}, meta, updated=True
            )

            # Only refine where refine_mask is true
            y = super().call_operator(
                op_where, (refine_mask, y_safe, y), {}, meta, updated=True
            )

        y0 = y

        # ---- special cases: NaN for |x|>1, +/-inf for |x|==1 ----
        nan = super().call_operator(
            op_div_t, (zeros, zeros), {}, meta, updated=True
        )  # 0/0
        pos_inf = super().call_operator(op_div_t, (ones, zeros), {}, meta, updated=True)
        inf_signed = super().call_operator(
            op_mul_t, (s, pos_inf), {}, meta, updated=True
        )

        out = y0
        out = super().call_operator(
            op_where, (abs_gt1, nan, out), {}, meta, updated=True
        )
        out = super().call_operator(
            op_where, (abs_eq1, inf_signed, out), {}, meta, updated=True
        )
        return out
