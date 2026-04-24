# BayesMCA — Bayesian Metabolic Control Analysis

BayesMCA is a Python package for **Bayesian inference of metabolic control properties** from multi-omics data. It uses linear-logarithmic (lin-log) kinetic models to connect enzyme expression, metabolite concentrations, and metabolic fluxes within a probabilistic framework, enabling the estimation of elasticity coefficients and metabolic control coefficients with full uncertainty quantification.

This project is a maintained fork of [pstjohn/emll](https://github.com/pstjohn/emll), updated to use **PyTensor** (successor to Theano) and **PyMC v5**.

## Key Features

- **Lin-log kinetic models** — Approximate enzyme kinetics as linear functions of log-transformed metabolite concentrations for efficient steady-state computation.
- **Bayesian inference** — Integrate proteomics, metabolomics, and fluxomics data into a unified probabilistic model using [PyMC](https://www.pymc.io/).
- **Control analysis** — Compute metabolite and flux control coefficients with uncertainty estimates.
- **Multiple solvers** — Choose from link-matrix, least-norm, Tikhonov-regularized, or pseudoinverse-based approaches depending on model structure.
- **COBRA integration** — Build elasticity matrices directly from [COBRApy](https://opencobra.github.io/cobrapy/) genome-scale metabolic models.

## Publications

Works using this codebase:
- [Bayesian Inference of Metabolic Kinetics from Genome-Scale Multiomics Data](https://dx.plos.org/10.1371/journal.pcbi.1007424) — St. John et al., *PLOS Computational Biology*, 2019
- [Bayesian Inference for Integrating Yarrowia lipolytica Multiomics Datasets with Metabolic Modeling](https://pubs.acs.org/doi/full/10.1021/acssynbio.1c00267) — ACS Synthetic Biology, 2021

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and packaging.

Install directly from GitHub:

```shell
uv pip install git+https://github.com/BioJazz-lab/BayesMCA.git
```

Or install in developer (editable) mode:

```shell
git clone https://github.com/BioJazz-lab/BayesMCA.git
cd BayesMCA
uv pip install -e .
```

You can also use pip if you prefer:

```shell
pip install git+https://github.com/BioJazz-lab/BayesMCA.git
```

Verify the installation:

```shell
python -c "import bayesmca"
```

### Dependencies

Core dependencies are installed automatically (see `pyproject.toml`). For the full
development/notebook environment, use the provided conda environment file:

```shell
conda env create -f environment.yml
```

## Package Structure

| Module | Description |
|---|---|
| `bayesmca.linlog_model` | Lin-log steady-state solvers (`LinLogLeastNorm`, `LinLogTikhonov`, `LinLogLinkMatrix`, etc.) |
| `bayesmca.pytensor_utils` | Differentiable PyTensor operations for linear solves with gradient support |
| `bayesmca.util` | Elasticity matrix construction, stoichiometric reduction, and Bayesian prior initialization |
| `bayesmca.data_model_integration` | Helpers for linking experimental observations to model predictions in PyMC |
| `bayesmca.test_models` | Reference COBRA models for testing and benchmarking |

## Quick Start

```python
import bayesmca
from bayesmca.util import create_elasticity_matrix, create_Ey_matrix, initialize_elasticity
import cobra

# Load a metabolic model
model = cobra.io.load_json_model("my_model.json")
N = cobra.util.create_stoichiometric_matrix(model)
Ex = create_elasticity_matrix(model)
Ey = create_Ey_matrix(model)
v_star = model.optimize().fluxes.values

# Create a lin-log model instance
ll = bayesmca.LinLogLeastNorm(N, Ex, Ey, v_star)

# Compute steady-state for perturbed enzyme levels
x_ss, v_ss = ll.steady_state_mat(en=perturbed_enzyme_levels)
```

See the `notebooks/` directory for complete worked examples including Bayesian inference with PyMC.

## How to Cite

If you use BayesMCA in your work, please cite:

```bibtex
@article{St.John2019,
  author    = {{St. John}, Peter C. and Strutz, Jonathan and Broadbelt, Linda J. and Tyo, Keith E. J. and Bomble, Yannick J.},
  doi       = {10.1371/journal.pcbi.1007424},
  editor    = {Maranas, Costas D.},
  issn      = {1553-7358},
  journal   = {PLOS Computational Biology},
  month     = {nov},
  number    = {11},
  pages     = {e1007424},
  title     = {{Bayesian inference of metabolic kinetics from genome-scale multiomics data}},
  url       = {https://dx.plos.org/10.1371/journal.pcbi.1007424},
  volume    = {15},
  year      = {2019}
}
```

## License

This project is licensed under the LGPL/GPL v2+ license. See [LICENSE](LICENSE) for details.
