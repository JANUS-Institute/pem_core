import numpy as np
from typing import TypeVar, Callable, Any
from numpy.typing import NDArray, ArrayLike
from amisc import System, distribution as distributions
import amisc.distribution
from MCMCIterators.samplers import DelayedRejectionAdaptiveMetropolis
from pem_core import Variable, PathLike, PEM
import math
import random
import sys
from pathlib import Path
import pandas as pd

T = TypeVar("T", bound=np.floating)
F = TypeVar("F", np.floating, float)

ID_HEADER = "id"
LOGPDF_HEADER = "log posterior"
ACCEPT_HEADER = "accepted"
COV_FILE = "cov_chol.csv"
SAMPLE_FILE = "samples.csv"

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

    return distance, likelihood

def _log_prior(system: System, params: dict[str, ArrayLike]) -> float:
    """
    Compute the log-prior distribution of a system when evaluated at a dictionary of parameter values
    """
    logp = 0.0
    for key, value in params.items():
        var = system.inputs()[key]
        assert isinstance(var, amisc.Variable)

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


class Sampler:
    variables: list[Variable]
    base_vars: dict[Variable, Any]
    init_sample_file: PathLike | None
    init_cov_file: PathLike | None
    system: PEM

    def __init__(
        self,
        variables,
        data,
        system,
        base_vars,
        opts,
        log_likelihood,
        init_sample_file: PathLike | None = None,
        init_cov_file: PathLike | None = None,
    ):
        self.variables = variables
        self.data = data
        self.system = system
        self.base_vars = base_vars
        self.opts = opts
        self.init_sample_file = init_sample_file
        self.init_cov_file = init_cov_file
        self.logpdf = lambda x: _log_posterior(system, dict(zip(variables, x)), log_likelihood)
        self._init_sample = None
        self._init_cov = None

    def cov(self):
        return self.initial_cov()

    def initial_sample(self):
        if self._init_sample is None:
            # Read initial sample from file or create it using the nominal values in the base_vars dict
            if self.init_sample_file is None:
                print("Constructing initial sample from variable nominal values")
                self._init_sample = np.array([self.base_vars[p] for p in self.variables])
            else:
                print(f"Reading initial sample from {self.init_sample_file}")
                var_names = [var.name for var in self.variables]
                self._init_sample = pd.read_csv(self.init_sample_file)[var_names].to_numpy()[-1, :]

        return self._init_sample

    def initial_cov(self):
        if self._init_cov is None:
            # No initial covariance matrix was provided, so we either construct one or read one from a file, if provided.
            if self.init_cov_file is None:
                print("Constructing initial covariance matrix from variable distributions")
                # Construct a diagonal covariance matrix based on the prior distributions of the variables
                variances = np.ones(len(self.variables))

                for i, p in enumerate(self.variables):
                    var = self.system.inputs()[p]
                    assert isinstance(var, amisc.Variable)
                    dist = var.distribution
                    if isinstance(dist, distributions.Uniform) or isinstance(dist, distributions.LogUniform):
                        lb, ub = dist.dist_args
                        std: float = (ub - lb) / 4
                    elif isinstance(dist, distributions.Normal):
                        std: float = dist.dist_args[1]
                    elif isinstance(dist, distributions.LogNormal):
                        std: float = dist.base ** dist.dist_args[1]
                    else:
                        raise ValueError(
                            f"Unsupported distribution {dist}. Currently only `Uniform`, `LogUniform`, `Normal` and `LogNormal` are supported."  # noqa: E501
                        )
                    # TODO: fix this typing issue
                    variances[i] = self.system.inputs()[p].normalize(std) ** 2  # type: ignore
                self._init_cov = np.diag(variances)
            else:
                print(f"Reading initial cov file from {self.init_cov_file}")
                # Construct covariance from file
                # The covariance in the file is saved as a lower triangular matrix from a Cholesky decomposition
                # Reconstruct the full square matrix
                var_names = [var.name for var in self.variables]
                df_chol = pd.read_csv(self.init_cov_file)
                cov_chol = df_chol.to_numpy().astype(np.float64)
                cov_matrix = cov_chol @ cov_chol.T

                # We support the variables being in a different order than in the original file, so we need to reconstruct and reindex a dataframe
                # TODO: this should work if we remove a variable, but if we add a variable this should respond gracefully.
                # Probably, we should default-construct an initial sample and covariace and then populate field-by-field based on the contents of the file.
                df_cov = pd.DataFrame(cov_matrix, columns=df_chol.columns)
                cov_matrix = df_cov[var_names].to_numpy().astype(np.float64)
                assert cov_matrix.shape[0] == cov_matrix.shape[1]

                self._init_cov = cov_matrix

            # Verify that the covariance matrix is positive-definite before proceeding.
            # This will throw an exception if not.
            self.init_cov = np.linalg.cholesky(self._init_cov)

        return self._init_cov

class PriorSampler(Sampler):
    """
    Samples from prior distribution, run model, and evaluates posterior probability
    """
    def __init__(
        self,
        variables,
        data,
        system,
        base_vars,
        opts,
        log_likelihood,
        init_sample_file: PathLike | None = None,
        init_cov_file: PathLike | None = None,
    ):
        super().__init__(variables, data, system, base_vars, opts, log_likelihood, init_sample_file, init_cov_file)

    def __iter__(self):
        return self

    def __next__(self):
        # TODO: fix this typing issue
        sample = np.array([var.normalize(var.distribution.sample((1,)))[0] for var in self.variables]) # type: ignore
        logp = self.logpdf(sample)
        return sample, logp, np.isfinite(logp)


class PreviousRunSampler(Sampler):
    """
    Samples (with replacement) from a previous MCMC run.
    """
    def __init__(
        self,
        variables,
        data,
        system,
        base_vars,
        opts,
        log_likelihood,
        prev_run_file: PathLike,
        init_sample_file: PathLike | None = None,
        init_cov_file: PathLike | None = None,
        burn_fraction: float = 0.5,
    ):
        super().__init__(variables, data, system, base_vars, opts, log_likelihood,init_sample_file, init_cov_file)

        # Read sample file from a previous run, discarding burned samples
        df = pd.read_csv(prev_run_file)
        num_samples = len(df[ACCEPT_HEADER])

        if burn_fraction > 0:
            num_burn = math.floor(burn_fraction * num_samples)
            df = df[:, num_burn:]

        var_names = [v.name for v in variables]
        self.samples = df[var_names].to_numpy()
        self.accepted = df[ACCEPT_HEADER]

    def __iter__(self):
        return self

    def sample_index(self):
        return random.randint(0, self.samples.shape[0] - 1)

    def __next__(self):
        # draw until we get an accepted sample
        index = self.sample_index()

        while not self.accepted[index]:
            index = self.sample_index()

        sample = self.samples[index, :]
        logp = self.logpdf(sample)
        return sample, logp, np.isfinite(logp)


class DRAMSampler(Sampler):
    """
    Samples using delayed rejection adaptive metropolis
    """
    def __init__(
        self,
        variables: list[amisc.Variable],
        data,
        system,
        base_vars,
        opts,
        log_likelihood,
        init_sample_file: PathLike | None = None,
        init_cov_file: PathLike | None = None,
    ):
        super().__init__(variables, data, system, base_vars, opts, log_likelihood, init_sample_file, init_cov_file)

        self.sampler = DelayedRejectionAdaptiveMetropolis(
            self.logpdf,
            self.initial_sample(),
            self.initial_cov(),
            adapt_start=10,
            eps=1e-6,
            sd=None,
            interval=1,
            level_scale=1e-1,
        )

    # TODO: fix this typing issue
    def cov(self):  # type: ignore
        return self.sampler.cov_chol

    def __iter__(self):
        return iter(self.sampler)

class SampleLogger:
    def __init__(self, sampler: Sampler, output_dir: PathLike | None = None, stream=sys.stdout, delimiter=","):
        self.best_logpdf = -np.inf
        self.samples = []
        self.logpdf = []
        self.accepted = []
        self.accept_num = 0
        self.cov = None
        self.output_dir = output_dir
        self.sample_file = None
        self.sampler = sampler
        self.stream = stream
        self.delimiter = delimiter
        self.sample_index = -1
        self.update(log=True)

        # Create sample directory if output_dir provided
        if self.output_dir is not None:
            self.output_dir = Path(self.output_dir)
            self.sample_file = self.output_dir / SAMPLE_FILE 
            print(f"Creating log file at {self.sample_file}")
    
            header_cols = [ID_HEADER] + [var.name for var in self.sampler.variables] + [LOGPDF_HEADER, ACCEPT_HEADER]
            header = delimiter.join(header_cols)

            with open(self.sample_file, "w") as fd:
                print(header, file=fd)

    def update(self, log=False):
        if isinstance(self.sampler, DRAMSampler):
            logp = self.sampler.sampler.current_logpdf
            self.best_logpdf = max(self.best_logpdf, logp)
            self.samples.append(self.sampler.sampler.current_sample)
            self.logpdf.append(logp)

            if self.sampler.sampler.accept_num > self.accept_num:
                self.accepted.append(True)
            else:
                self.accepted.append(False)

            self.accept_num = self.sampler.sampler.accept_num
            self.cov = self.sampler.sampler.cov_chol

        self.sample_index = len(self.samples) - 1

        if log:
            self.log()

            if self.sample_file is not None:
                self.update_files()

    def log(self):
        print(
            f"sample: {self.sample_index}, logp: {self.logpdf[-1]:.3f}, best logp: {self.best_logpdf:.3f},",
            f"accepted: {self.accepted[-1] if self.accepted else False}, p_accept: {self.accept_num / len(self.samples) * 100:.1f}%",
            file=self.stream
        )
    
    def update_files(self):
        assert self.output_dir is not None
        assert self.sample_file is not None
        assert self.sample_index >= 0

        # Write log file
        id_str = f"{len(self.samples):06d}"
        with open(self.sample_file, "a") as fd:
            row = [id_str] + [f"{s}" for s in self.samples[-1]] + [f"{self.logpdf[-1]}", f'{self.accepted[-1]}']
            print(self.delimiter.join(row), file=fd)

        # Write covariance matrix
        if isinstance(self.sampler, DRAMSampler):
            var_names = [var.name for var in self.sampler.variables]
            df = pd.DataFrame(self.sampler.sampler.cov_chol, columns=var_names)
            df.to_csv(Path(self.output_dir) / COV_FILE, index=False)
