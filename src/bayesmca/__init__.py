"""BayesMCA: Bayesian Metabolic Control Analysis.

A Python package for Bayesian inference of metabolic kinetics using
linear-logarithmic (lin-log) kinetic models. BayesMCA enables the integration
of multi-omics data (metabolomics, proteomics, fluxomics) with stoichiometric
metabolic models to infer elasticity coefficients and control coefficients
using Bayesian statistical methods.

This package provides:
    - Lin-log kinetic model solvers for steady-state metabolite concentrations and fluxes
    - PyTensor-based operations for integration with PyMC probabilistic models
    - Utilities for constructing elasticity matrices from COBRA metabolic models
    - Stoichiometric reduction methods (Smallbone, Waldherr) for numerical stability
"""

from bayesmca.linlog_model import *  # noqa: F401, F403
from bayesmca.util import create_elasticity_matrix, create_Ey_matrix  # noqa: F401

import bayesmca.test_models  # noqa: F401
