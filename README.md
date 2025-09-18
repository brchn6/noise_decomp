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
## References

* Elowitz MB, Levine AJ, Siggia ED, Swain PS. Stochastic Gene Expression in a Single Cell. Science (2002).

* Raser JM, O’Shea EK. Control of Stochasticity in Eukaryotic Gene Expression. Science (2004).

