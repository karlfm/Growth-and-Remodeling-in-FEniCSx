# G&R Solver

A FEniCSx solver for growth and remodeling.

## Installation

To run this program you need the dolfinx Docker image:
```bash
docker run -ti -v $(pwd):/home/shared dolfinx/dolfinx:stable
```

## Use

Run `comparison_script.py` to reproduce the plots in the paper. See the `GrowthLaws/` directory for examples.
