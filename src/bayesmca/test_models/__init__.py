"""Reference metabolic models for BayesMCA testing and benchmarking.

Provides a collection of COBRA metabolic models of varying complexity used
for validating lin-log model solvers, control coefficient calculations, and
Bayesian inference pipelines. Models range from simple textbook examples to
genome-scale reconstructions.

Available models (via the ``models`` dictionary):
    - teusink: Yeast glycolysis model (Teusink et al.)
    - mendes: Small example pathway (Mendes et al.)
    - textbook: Reduced E. coli core model
    - greene_small: 4-metabolite, 6-reaction toy network
    - greene_large: 16-metabolite, 26-reaction network
    - contador: Corynebacterium glutamicum lysine production model
    - jol2012: Trimmed S. cerevisiae genome-scale model
"""

import os
import cobra
import numpy as np

import pandas as pd

from cobra.util.array import create_stoichiometric_matrix

currdir = os.path.dirname(os.path.abspath(__file__))

def get_N_v(model):
    """Compute the stoichiometric matrix and reference flux vector from a COBRA model.

    Optimizes the model via FBA, then orients all reactions so that reference
    fluxes are non-negative (flipping stoichiometry columns for reverse fluxes).

    Parameters
    ----------
    model : cobra.Model
        A COBRA metabolic model with a defined objective.

    Returns
    -------
    N : np.ndarray
        The (n_metabolites x n_reactions) stoichiometric matrix with columns
        oriented for non-negative reference fluxes.
    v_star : np.ndarray
        The reference steady-state flux vector (all entries >= 0).
    """

    solution = model.optimize()

    N = create_stoichiometric_matrix(model)
    v_star = solution.fluxes.values

    # Flip columns for reactions with negative flux so all v_star entries >= 0
    for i, v in enumerate(v_star):
        if v < 0:
            N[:, i] *= -1
            v_star[i] *= -1

    assert np.all(v_star >= 0)

    return N, v_star


def load_contador():
    """Load the Contador C. glutamicum lysine production model."""
    model = cobra.io.load_json_model(currdir + '/contador.json')
    model.reactions.EX_glc.bounds = (-1.243, 1000)
    model.reactions.EX_lys.lower_bound = .139
    model.reactions.zwf.lower_bound = .778

    N, v_star = get_N_v(model)

    return model, N, v_star

def load_teusink():
    """Load the Teusink yeast glycolysis model (BIOMD0000000064)."""
    model = cobra.io.read_sbml_model(currdir + '/BIOMD0000000064.xml')
    model.reactions.vGLT.bounds = (-88.1, 88.1)
    for rxn in model.reactions:
        rxn.lower_bound = 0.1

    model.objective = model.reactions.vATP

    N, v_star = get_N_v(model)

    return model, N, v_star

def load_mendes():
    """Load the Mendes small example pathway model."""
    from .mendes_model import model
    N = create_stoichiometric_matrix(model)
    v_star = np.array([0.0289273 ,  0.0289273 ,  0.01074245,  0.01074245,  0.01074245,
                       0.01818485,  0.01818485,  0.01818485])
    return model, N, v_star

def load_textbook():
    """Load the reduced E. coli textbook model."""

    model = cobra.io.load_json_model(currdir + '/textbook_reduced.json')
    N, v_star = get_N_v(model)

    return model, N, v_star

def load_greene_small():
    """Load the Greene small toy network (4 metabolites, 6 reactions)."""

    N = np.array([
        [-1, 0, 0, 1, 0, 0],
        [1, 1, -1, 0, 0, 0],
        [1, -1, 0, 0, 0, -1],
        [0, 0, 1, 0, -1, 0]])

    v_star = np.array([1, 0.5, 1.5, 1, 1.5, 0.5])

    rxn_names = ['V1', 'V2', 'V3', 'Vin', 'Vout', 'Voutx3']
    met_names = ['x1', 'x2', 'x3', 'x4']

    assert np.allclose(N @ v_star, 0)

    return construct_model_from_mat(N, rxn_names, met_names), N, v_star


def load_greene_large():
    """Load the Greene large network (16 metabolites, 26 reactions)."""

    N = np.array([
        [-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0],
        [0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, -1, 0],
        [-1, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 1, -1, 0, 0],
        [1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -1, -1, 1, 0, 0]])

    v_star = np.array([1, 0, 1, 0, 1, 0, 0.5, 0, 0.5, 0, 0.5, 0, 1.5, 0, 1.5,
                       0, 1.5, 0, 1, 0, 1, 0, 1, 0, 1.5, 0.5])

    rxn_names = ['v1_1', 'v1_2', 'v1_3', 'v1_4', 'v1_5', 'v1_6', 'v2_1',
                 'v2_2', 'v2_3', 'v2_4', 'v2_5', 'v2_6', 'v3_1', 'v3_2', 'v3_3',
                 'v3_4', 'v3_5', 'v3_6', 'vin_1', 'vin_2', 'vin_3', 'vin_4',
                 'vin_5', 'vin_6', 'vout', 'voutx3']

    met_names = ['x1', 'x2', 'x3', 'x4', 'E1', 'E2', 'E3', 'Ein', 'x1E1',
                 'x3E1', 'x3E2', 'x2E2', 'x2E3', 'x4E3', 'xoutEin', 'xinEin']

    assert np.allclose(N @ v_star, 0)

    return construct_model_from_mat(N, rxn_names, met_names), N, v_star


def load_jol2012_edit():
    """Load the trimmed Jol 2012 S. cerevisiae genome-scale model."""

    model = cobra.io.load_json_model(currdir + '/jol2012_trimmed.json')
    v_star = pd.read_pickle(currdir + '/jol2012_vstar.p').values

    N = cobra.util.create_stoichiometric_matrix(model)

    assert np.allclose(N @ v_star, 0)

    return model, N, v_star


def construct_model_from_mat(N, rxn_names, met_names):
    """Construct a COBRA model from a stoichiometric matrix and name lists.

    Parameters
    ----------
    N : np.ndarray
        Stoichiometric matrix (n_metabolites x n_reactions).
    rxn_names : list of str
        Reaction identifiers.
    met_names : list of str
        Metabolite identifiers.

    Returns
    -------
    cobra.Model
        A COBRA model with the specified stoichiometry.
    """

    model = cobra.Model('test_model')

    model.add_metabolites([cobra.Metabolite(id=met_name) for met_name in met_names])

    for row, rxn_name in zip(N.T, rxn_names):
        reaction = cobra.Reaction(id=rxn_name)
        model.add_reactions([reaction])
        reaction.add_metabolites({
            met_id: float(stoich) for met_id, stoich in zip(met_names, row)
            if abs(stoich) > 1E-6})

    return model


models = {
    'teusink': load_teusink,
    'mendes': load_mendes,
    'textbook': load_textbook,
    'greene_small': load_greene_small,
    'greene_large': load_greene_large,
    'contador': load_contador,
    'jol2012': load_jol2012_edit,
}
