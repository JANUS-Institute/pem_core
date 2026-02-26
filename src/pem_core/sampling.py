"""
Bayesian sampling tools for PEM (Predictive Engineering Model) calibration.

Provides utilities for Markov Chain Monte Carlo (MCMC)-based inference over
PEM input parameters, including:

- Log-likelihood helpers: Gaussian 1D log-PDF and relative L2-norm-based
  likelihood functions for comparing model output to observations.
- Log-posterior computation: combines a prior (from variable distributions
  defined in the PEM) with a user-supplied log-likelihood function.
- Sampler base class (`Sampler`): abstract iterator that manages sample
  state, acceptance tracking, file I/O (CSV sample log and Cholesky
  covariance), and screen logging. Supports warm-starting from previous
  runs via `init_sample_file` and `init_cov_file`.
- Concrete samplers:
    - `PriorSampler`: draws independent samples from the prior distribution.
    - `PreviousRunSampler`: resamples (with replacement) from accepted
      samples of a prior MCMC run, useful for certain workflows.
    - `DRAMSampler`: uses the Delayed Rejection Adaptive Metropolis (DRAM)
      algorithm for efficient posterior exploration.
"""
from abc import ABC, abstractmethod
import itertools
import math
import os
from pathlib import Path
import random
import sys
from typing import Callable, TypeVar

import numpy as np
from numpy.typing import NDArray
import pandas as pd

from amisc import distribution as distributions
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis
from pem_core import ArrayLike, PathLike, PEM, Variable

# Type aliases
T = TypeVar("T", bound=np.floating)
F = TypeVar("F", np.floating, float)
LikelihoodType = Callable[[PEM, dict[str, ArrayLike], PathLike | None], float]

# Constants defining output file names and column headers
ID_HEADER = "id"
LOGPDF_HEADER = "log posterior"
ACCEPT_HEADER = "accepted"
COV_FILE = "cov_chol.csv"
SAMPLE_FILE = "samples.csv"

def gauss_logpdf_1D(x: F, mean: F, std: F) -> F:
    """
    Gaussian log-likelihood in 1D
    """
    return -0.5 * (np.log(2 * np.pi * std**2) + (x - mean) ** 2 / std**2)

def relative_l2_norm(data: NDArray[T], observation: NDArray[T]) -> T:
    """
    Compute the L2-normed distance between a data array and observation array,
    relative to the magnitude of the data array.
    """
    num = np.sum((data - observation) ** 2)
    denom = np.sum(data**2)
    return np.sqrt(num / denom)

def relative_gaussian_likelihood(x: ArrayLike, y: ArrayLike, std: float) -> tuple[float, float]:
    """
    Calculate the gaussian likelihood of the L2 norm of the distance between x and y if the norm is half-normally distributed about zero with standard deviation `std`. 
    Returns the likelihood and the L2 distance
    """
    x_arr = np.asarray(x)
    data_arr = np.asarray(y)
    if x_arr.shape != data_arr.shape:
        raise ValueError(f"Shape of x must match shape of data! Got {x_arr.shape} and {data_arr.shape} instead.")
    
    distance = relative_l2_norm(x_arr, data_arr)

    # Since the l2-norm is positive-definite, the distribution is a half-normal, so we
    # need to multiply the normal pdf by 2 (or add ln(2) to the log-pdf)
    likelihood = gauss_logpdf_1D(distance, 0.0, std) + np.log(2)

    return distance, likelihood

def _log_prior(pem: PEM, params: dict[str, ArrayLike]) -> float:
    """
    Compute the log-prior distribution of a pem when evaluated at a dictionary of parameter values
    """
    logp = 0.0
    for key, value in params.items():
        var = pem.inputs()[key]

        # TODO: fix amisc typing so I can remove these type assertions
        assert isinstance(var, Variable)
        assert var.distribution is not None and isinstance(var.distribution, distributions.Distribution)

        denorm = var.denormalize(value)
        prior = var.distribution.pdf(np.asarray(denorm))

        if isinstance(prior, np.ndarray):
            prior = prior[0]

        if prior <= 0:
            return -np.inf

        logp += np.log(prior)

    return logp

def _log_posterior(pem: PEM, params: dict[str, ArrayLike], likelihood: LikelihoodType, stats: dict) -> float:
    """
    Compute the log posterior distribution by adding the log-prior distribution of the PEM and user-provided log-likelihood distribution.
    Returns -inf if either evaluate to something non-finite.
    """
    log_prior = _log_prior(pem, params)

    if not np.isfinite(log_prior):
        return -np.inf

    log_likelihood = likelihood(pem, params, stats["current_sample_dir"])

    if not np.isfinite(log_likelihood):
        return -np.inf

    return log_prior + log_likelihood

class Sampler(ABC):
    def __init__(
        self,
        pem: PEM,
        sample_vars: list[Variable],
        base_vars: dict[Variable, str],
        log_likelihood: LikelihoodType,
        stream = sys.stdout,
        output_dir: PathLike | None = None,
        init_sample_file: PathLike | None = None,
        init_cov_file: PathLike | None = None,
    ):
        self.pem = pem
        self.sample_vars = sample_vars
        self.base_vars = base_vars
        self.init_sample_file = init_sample_file
        self.init_cov_file = init_cov_file
        self.variable_names = [v.name for v in self.sample_vars]

        # Sampler stats
        self.current_logpdf = -np.inf
        self.accepted = False
        self.current_sample = self.initial_sample()
        self.current_cov = self.initial_cov()
        self.best_sample = self.current_sample
        self.best_logpdf = self.current_logpdf
        self.accept_num = 0
        self.sample_num = 0
        self.p_accept = 0.0

        # Output files
        self.stream = stream
        self.output_dir = output_dir
        self.output_delimiter = ","

        # Create sample file if requested
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
            self.sample_file = self.output_dir / SAMPLE_FILE 

            print(f"Creating log file at {self.sample_file}")
            header_cols = [ID_HEADER] + self.variable_names + [LOGPDF_HEADER, ACCEPT_HEADER]
            header = self.output_delimiter.join(header_cols)

            with open(self.sample_file, "w") as fd:
                print(header, file=fd)

            self.sample_dir = self.output_dir / "samples"
            os.mkdir(self.sample_dir)
        else:
            self.sample_dir = None
            self.sample_file = None

        # Logpdf
        self.sampler_dict = dict(current_sample_dir=self._current_sample_dir(), stream=self.stream)
        self.logpdf = lambda x: _log_posterior(pem, dict(zip(self.variable_names, x)), log_likelihood, self.sampler_dict)

    def _current_sample_dir(self):
        if self.sample_dir is None:
            return None
        else:
            return self.sample_dir / f"{self.sample_num:06d}"

    def log_to_screen(self):
        print(
            f"sample: {self.sample_num - 1}, logp: {self.current_logpdf:.3f}, best logp: {self.best_logpdf:.3f},",
            f"accepted: {self.accepted}, p_accept: {self.p_accept * 100:.1f}%",
            file=self.stream
        )

    def write_files(self):
        assert self.output_dir is not None
        assert self.sample_file is not None
        assert self.sample_num >= 1

        # Write log file
        id_str = f"{self.sample_num - 1:06d}"
        with open(self.sample_file, "a") as fd:
            row = [id_str] + [f"{s}" for s in self.current_sample] + [f"{self.current_logpdf}", f'{self.accepted}']
            print(self.output_delimiter.join(row), file=fd)

        # Write covariance matrix
        df = pd.DataFrame(self.current_cov, columns=self.variable_names)
        df.to_csv(Path(self.output_dir) / COV_FILE, index=False)

    def initial_sample(self):
        """
        Read initial sample from file or create it using the nominal values in the base_vars dict
        """
        if self.init_sample_file is None:
            print("Constructing initial sample from variable nominal values")
            sample = np.array([self.base_vars[p] for p in self.sample_vars])
        else:
            print(f"Reading initial sample from {self.init_sample_file}")
            sample = pd.read_csv(self.init_sample_file)[self.variable_names].to_numpy()[-1, :]

        return sample

    def initial_cov(self):
        """
        Read a starting covariance matrix from a file or create it using distributions of the calibration variables.
        """
        if self.init_cov_file is None:
            print("Constructing initial covariance matrix from variable distributions")
            # Construct a diagonal covariance matrix based on the prior distributions of the variables
            variances = np.ones(len(self.sample_vars))

            for i, var in enumerate(self.sample_vars):
                assert isinstance(var, Variable)
                dist = var.distribution
                if isinstance(dist, distributions.Uniform) or isinstance(dist, distributions.LogUniform):
                    lb, ub = dist.dist_args
                    std: float = (ub - lb) / 4
                elif isinstance(dist, distributions.Normal):
                    std: float = dist.dist_args[1]
                elif isinstance(dist, distributions.LogNormal):
                    std: float = dist.base ** dist.dist_args[1]
                else:
                    raise ValueError(f"Unsupported distribution {dist}. Currently only `Uniform`, `LogUniform`, `Normal` and `LogNormal` are supported.")
                variances[i] = var.normalize(std) ** 2
            cov = np.diag(variances)
        else:
            print(f"Reading initial cov file from {self.init_cov_file}")
            # Construct covariance from file
            # The covariance in the file is saved as a lower triangular matrix from a Cholesky decomposition
            # Reconstruct the full square matrix
            df_chol = pd.read_csv(self.init_cov_file)
            cov_chol = df_chol.to_numpy().astype(np.float64)
            cov_matrix = cov_chol @ cov_chol.T

            # We support the variables being in a different order than in the original file, so we need to reconstruct and reindex a dataframe
            # TODO: this should work if we remove a variable, but if we add a variable this should respond gracefully.
            # Probably, we should default-construct an initial sample and covariace and then populate field-by-field based on the contents of the file.
            df_cov = pd.DataFrame(cov_matrix, columns=df_chol.columns)
            cov_matrix = df_cov[self.variable_names].to_numpy().astype(np.float64)
            assert cov_matrix.shape[0] == cov_matrix.shape[1]
            cov = cov_matrix

        # Verify that the covariance matrix is positive-definite before proceeding.
        # This will throw an exception if not.
        return np.linalg.cholesky(cov)

    def update_stats(self, sample, logp, accepted):
        """
        Update and log sampler stats, including acceptance percentage and the best sample so far.
        If requested, writes sample and covariance files, as well as logs a summary to the screen.
        """
        if logp > self.current_logpdf:
            self.best_logpdf, self.best_sample = logp, sample

        if accepted:
            self.accept_num += 1
        
        self.sample_num += 1
        self.sampler_dict["current_sample_dir"] = self._current_sample_dir()
        self.p_accept = float(self.accept_num) / float(self.sample_num)
        self.current_sample, self.current_logpdf, self.accepted = sample, logp, accepted

        if self.stream is not None:
            self.log_to_screen()

        if self.output_dir is not None:
            self.write_files()

    def cov(self):
        """Convert covariance matrix from lower-triangular Cholesky form to full form on request"""
        return self.current_cov @ self.current_cov.T

    @abstractmethod
    def propose_sample(self) -> tuple[ArrayLike, float, bool]:
        pass

    def __iter__(self):
        return self

    def __next__(self):
        sample, logp, accepted = self.propose_sample()
        self.update_stats(sample, logp, accepted)
        return sample, logp, accepted

    def sample(self, num_samples: int) -> list[tuple[ArrayLike, float, bool]]:
        return [s for s in itertools.islice(self, num_samples)]

class PriorSampler(Sampler):
    """
    Samples from prior distribution, run model, and evaluates posterior probability
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__next__()

    def propose_sample(self):
        # TODO: fix this typing issue
        sample = np.array([var.normalize(var.distribution.sample((1,)))[0] for var in self.sample_vars]) # type: ignore
        logp = self.logpdf(sample)
        return sample, logp, np.isfinite(logp)

class PreviousRunSampler(Sampler):
    """
    Samples (with replacement) from a previous MCMC run.
    """
    def __init__(self, prev_run_file, *args, burn_fraction=0.5, **kwargs):
        super().__init__(*args, **kwargs)

        # Read sample file from a previous run, discarding burned samples
        df = pd.read_csv(prev_run_file)
        num_samples = len(df[ACCEPT_HEADER])

        if burn_fraction > 0:
            num_burn = math.floor(burn_fraction * num_samples)
            df = df.iloc[num_burn:]

        self.prev_samples = df[self.variable_names].to_numpy()
        self.prev_accepted = df[ACCEPT_HEADER].to_numpy()

        # Draw an initial sample
        self.__next__()

    def _sample_index(self):
        # Draw until we get a sample that was accepted in our previous run
        while True:
            index = random.randint(0, self.prev_samples.shape[0] - 1)
            if self.prev_accepted[index]:
                return index

    def propose_sample(self):
        index = self._sample_index()
        sample = self.prev_samples[index, :]
        logp = self.logpdf(sample)
        return sample, logp, np.isfinite(logp)

class DRAMSampler(Sampler):
    """Samples using the Delayed Rejection Adaptive Metropolis (DRAM) algorithm"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sampler = DelayedRejectionAdaptiveMetropolis(
            self.logpdf, self.current_sample, self.cov(),
            adapt_start=10, eps=1e-6, sd=None, interval=1, level_scale=1e-1,
        )

        self.update_stats(self.sampler.current_sample, self.sampler.current_logpdf, self.sampler.accept_num > 0)

    def propose_sample(self):
        return self.sampler.__next__()