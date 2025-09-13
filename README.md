# PyDPMD

Python data classes mirroring DPMD particle classes.

Greatly facilitates data pre (particle class initializations) and post (efficient and general time-correlation function calculation) processing as well as multi-system operations, arguably the most useful feature of DPMD.

## Installation

```bash
pip install -e .
```

## Usage

```python
from pydpmd.utils import join_systems
from pydpmd.data import RigidBumpy
from pydpmd.calc import run_binned, LagBinsLog, msd_kernel

rb = RigidBumpy()  # initialize the system of rigid bumpy particles
joined = join_systems([rb for _ in range(10000)])  # batch 10k systems together, effectively running 10k jobs at once

# run some trajectory-generating process

bins = LagBinsLog.from_source(joined.trajectory, num_per_decade=10)  # calculate msd over log-spaced time-lags
res = run_binned(msd_kernel, joined.trajectory, bins, kernel_kwargs={'system_id': joined.system_id, 'system_size': joined.system_size}, show_progress=True,)
t = bins.values()  # (T,) array of time-lags
msd = res.mean  # (T, N_systems,) array of msds for each system
```

## Development

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest
```
