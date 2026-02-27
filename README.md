# pem_core

Core library for the Predictive Engineering Model (PEM) developed for the [NASA JANUS STRI](https://januselectricpropulsion.com/).

`pem_core` wraps the [`amisc`](https://github.com/eckelsjd/amisc) `System` framework to construct physics-based models of electric propulsion systems, adds utilities for loading and standardizing experimental CSV data, and provides Bayesian (MCMC) tools for parameter calibration against measured data.

---

## Installation

Requires Python >= 3.11.

```bash
pip install -e .
```

---

## Modules

### `pem_core.PEM`

Extends `amisc.System`. Load and run a multi-component PEM with components defined in an `amisc` YAML configuration.

```python
from pem_core import PEM

pem = PEM.from_file("model.yaml")          # load from YAML config
pem = PEM.from_directory("run_dir/")       # auto-discover first .yaml/.yml in a directory

nominals = pem.get_nominal_inputs(norm=True)            # dict of normalized nominal inputs
calib_vars = pem.get_inputs_by_category("calibration")  # list[Variable], sorted by name

result = pem(inputs)
```
---

### `pem_core.data`

Utilities for loading experimental data from CSV files into a structured format that pairs quantities-of-interest (QoIs) with their operating conditions.
We use the xarray package to make handling of scalar and field quantities seamless.

### `pem_core.sampling`

Bayesian calibration and posterior sampling via MCMC, as well as auxilliary samplers for the prior distribution and previous MCMC runs.
All samplers share the same base interface (`Sampler`) and are python iterators:

```python
# Iterate one sample at a time
sample, logp, accepted = next(sampler)

# Or draw n samples at once
samples = sampler.sample(n)   # list of (sample, logp, accepted) tuples

# Draw samples in a loop:
for sample, logp, accepted in sampler:
    # do something with the sampler

```

**`PriorSampler`** — draws i.i.d. samples from the prior.

**`PreviousRunSampler`** — resamples (with replacement) from accepted samples of a previous MCMC run, discarding a configurable burn-in fraction.

**`DRAMSampler`** — Delayed Rejection Adaptive Metropolis algorithm for efficient posterior exploration.


## End-to-end calibration example

Below is a schematic example outlining all the major parts of using the PEM to calibrate an EP model system.
See the [HallThrusterPEM repo](https://github.com/JANUS-Institute/HallThrusterPEM) for a real-world example.

```python
from pem_core import PEM
from pem_core.data import load_single_dataset
from pem_core.sampling import DRAMSampler, relative_gaussian_likelihood

# 1. Load experimental data and declare which variables we treat as operating variables.
# Specifying units allows us to auto-convert columns.
# Units can be plain strings or pint.Unit objects — no need to import pint.
# See pem_core/data.py for a description of our data processing methodology.
operating_vars = {
    "discharge voltage": {"unit": "V"},
    "anode mass flow rate": {"unit": "mg/s"},
}

# Declare that we're looking for thrust
qois = {"thrust": {"unit": "mN"}}

dataset = load_single_dataset("experiment.csv", operating_vars, qois)

# 2. Load PEM model
pem = PEM.from_file("model.yaml", output_dir="run/")
calib_vars = pem.get_inputs_by_category("calibration")
nominals = pem.get_nominal_inputs(norm=True)

# 3. Define likelihood
def log_likelihood(pem, params, sample_dir):
    inputs = {**nominals, **params}
    outputs = pem(inputs)
    # Compare outputs to dataset ...
    # The specifics here depend on the data and how it is formatted, 
    # as well as which likelihood makes most sense for the problem
    sim_thrust = # ... process outputs
    data_thrust = # ... process/handle data

    _, logp = relative_gaussian_likelihood(sim_thrust, data_thrust, std=0.05)
    return logp

# 4. Run MCMC
sampler = DRAMSampler(
    pem=pem,
    sample_vars=calib_vars,
    base_vars={v: nominals[v.name] for v in calib_vars},
    log_likelihood=log_likelihood,
    output_dir="run/mcmc/",
)

samples = sampler.sample(5_000)
print(f"Acceptance rate: {sampler.p_accept:.1%}")
```
