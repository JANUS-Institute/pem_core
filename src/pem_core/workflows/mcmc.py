import numpy as np
from typing import TypeVar, Callable
from numpy.typing import NDArray, ArrayLike
from amisc import System
import amisc.distribution

T = TypeVar("T", bound=np.floating)
F = TypeVar("F", np.floating, float)

def _gauss_logpdf_1D(x: F, mean: F, std: F) -> F:
    """
    Gaussian log-likelihood in 1D
    """
    return -0.5 * (np.log(2 * np.pi * std**2) + (x - mean) ** 2 / std**2)

def _relative_l2_norm(data: NDArray[T], observation: NDArray[T]) -> T:
    """
    Compute the L2-normed distance between a data array and observation array,
    relative to the magnitude of the data array.
    """
    num = np.sum((data - observation) ** 2)
    denom = np.sum(data**2)
    return np.sqrt(num / denom)

def _relative_gaussian_likelihood(x: ArrayLike, y: ArrayLike, std: float) -> tuple[float, float]:
    """
    Calculate the gaussian likelihood of the L2 norm of the distance between x and y if the norm is half-normally distributed about zero with standard deviation `std`. 
    Returns the likelihood and the L2 distance
    """
    x_arr = np.asarray(x)
    data_arr = np.asarray(y)
    if x_arr.shape != data_arr.shape:
        raise ValueError(f"Shape of x must match shape of data! Got {x_arr.shape} and {data_arr.shape} instead.")
    
    distance = _relative_l2_norm(x_arr, data_arr)

    # Since the l2-norm is positive-definite, the distribution is a half-normal, so we
    # need to multiply the normal pdf by 2 (or add ln(2) to the log-pdf)
    likelihood = _gauss_logpdf_1D(distance, 0.0, std) + np.log(2)

    return likelihood, distance

def _log_prior(system: System, params: dict[str, ArrayLike]) -> float:
    """
    Compute the log-prior distribution of a system when evaluated at a dictionary of parameter values
    """
    logp = 0.0
    for key, value in params.items():
        var = system.inputs()[key]
        denorm = var.denormalize(value)

        assert var.distribution is not None and isinstance(var.distribution, amisc.distribution.Distribution)
        prior = var.distribution.pdf(np.asarray(denorm))

        if isinstance(prior, np.ndarray):
            prior = prior[0]

        if prior <= 0:
            return -np.inf

        logp += np.log(prior)

    return logp

def _log_posterior(system: System, params: dict[str, ArrayLike], likelihood: Callable[[System, dict[str, ArrayLike]], float]) -> float:

    log_prior = _log_prior(system, params)

    if not np.isfinite(log_prior):
        return -np.inf

    log_likelihood = likelihood(system, params)

    if not np.isfinite(log_likelihood):
        return -np.inf

    return log_prior + log_likelihood
