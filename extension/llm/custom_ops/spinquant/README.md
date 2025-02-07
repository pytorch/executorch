# SpinQuant

This is an implementation of the [Fast Hadamard
Transform](https://en.wikipedia.org/wiki/Fast_Walsh–Hadamard_transform)
as used in [SpinQuant](https://arxiv.org/abs/2405.16406) (for the R3
and R4 matrices), [QuaRot](https://arxiv.org/abs/2404.00456), and
[Quip#](https://arxiv.org/pdf/2402.04396). We follow those papers'
method (as implemented in
https://github.com/Dao-AILab/fast-hadamard-transform/) for extending
the transform to non-power-of-two input sizes. CUDA is not considered
because https://github.com/Dao-AILab/fast-hadamard-transform/ is
already available.

The intended long-term destination for this code is pytorch/ao; it is
in ExecuTorch temporarily until we get C++ dependency from ExecuTorch
on torchao figured out.
