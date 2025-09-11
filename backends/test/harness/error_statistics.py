from dataclasses import dataclass

import torch
from torch.ao.ns.fx.utils import compute_sqnr


@dataclass
class TensorStatistics:
    """Contains summary statistics for a tensor."""

    shape: torch.Size
    """ The shape of the tensor. """

    numel: int
    """ The number of elements in the tensor. """

    median: float
    """ The median of the tensor. """

    mean: float
    """ The mean of the tensor. """

    max: torch.types.Number
    """ The maximum element of the tensor. """

    min: torch.types.Number
    """ The minimum element of the tensor. """

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TensorStatistics":
        """Creates a TensorStatistics object from a tensor."""
        flattened = torch.flatten(tensor)
        return cls(
            shape=tensor.shape,
            numel=tensor.numel(),
            median=torch.quantile(flattened, q=0.5).item(),
            mean=flattened.mean().item(),
            max=flattened.max().item(),
            min=flattened.min().item(),
        )


@dataclass
class ErrorStatistics:
    """Contains statistics derived from the difference of two tensors."""

    reference_stats: TensorStatistics
    """ Statistics for the reference tensor. """

    actual_stats: TensorStatistics
    """ Statistics for the actual tensor. """

    error_l2_norm: float | None
    """ The L2 norm of the error between the actual and reference tensor. """

    error_mae: float | None
    """ The mean absolute error between the actual and reference tensor. """

    error_max: float | None
    """ The maximum absolute elementwise error between the actual and reference tensor. """

    error_msd: float | None
    """ The mean signed deviation between the actual and reference tensor. """

    sqnr: float | None
    """ The signal-to-quantization-noise ratio between the actual and reference tensor. """

    @classmethod
    def from_tensors(
        cls, actual: torch.Tensor, reference: torch.Tensor
    ) -> "ErrorStatistics":
        """Creates an ErrorStatistics object from two tensors."""
        actual = actual.to(torch.float64)
        reference = reference.to(torch.float64)

        if actual.shape != reference.shape:
            return cls(
                reference_stats=TensorStatistics.from_tensor(reference),
                actual_stats=TensorStatistics.from_tensor(actual),
                error_l2_norm=None,
                error_mae=None,
                error_max=None,
                error_msd=None,
                sqnr=None,
            )

        error = actual - reference
        flat_error = torch.flatten(error)

        return cls(
            reference_stats=TensorStatistics.from_tensor(reference),
            actual_stats=TensorStatistics.from_tensor(actual),
            error_l2_norm=torch.linalg.norm(flat_error).item(),
            error_mae=torch.mean(torch.abs(flat_error)).item(),
            error_max=torch.max(torch.abs(flat_error)).item(),
            error_msd=torch.mean(flat_error).item(),
            # Torch sqnr implementation requires float32 due to decorator logic
            sqnr=compute_sqnr(actual.to(torch.float), reference.to(torch.float)).item(),
        )
