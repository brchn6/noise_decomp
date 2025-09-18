# noise_decomp

A **super small** Python package that implements the dual-reporter
decomposition of **intrinsic** and **extrinsic** noise from paired
single-cell measurements.

## Install (editable)
```bash
pip install -e .
```

---
## Usage
``` python 
from noise_decomp import noise_decomp

res = noise_decomp(r, g)
print(res)

```

---
## CLI
After installing (editable) the package you get a small CLI `noise-decomp`:

Compute from comma-separated values:

```bash
noise-decomp --r 1,2,3 --g 1.1,2.1,3.1
```

Compute from files (CSV or whitespace separated):

```bash
noise-decomp --rfile path/to/r.csv --gfile path/to/g.csv
```

Use `--no-normalize` to disable mean normalization and `--ddof` to set the delta degrees of freedom for variance/covariance.

---
## Cython (fast) extension

To build the Cython extension for best performance:

```bash
pip install cython numpy
python build_cython.py
```

The Python API will automatically use the fast Cython backend if available.

---
## Examples
Run the tiny demo script:

```bash
python examples/demo.py
```

---
## References

* Elowitz MB, Levine AJ, Siggia ED, Swain PS. Stochastic Gene Expression in a Single Cell. Science (2002).

* Raser JM, O’Shea EK. Control of Stochasticity in Eukaryotic Gene Expression. Science (2004).

