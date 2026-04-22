"""Lin-log kinetic model definitions for BayesMCA.

This module implements the core linear-logarithmic (lin-log) kinetic model classes
used for computing steady-state metabolite concentrations and fluxes. The lin-log
framework approximates enzyme kinetics as linear functions of log-transformed
metabolite concentrations, enabling efficient Bayesian inference of metabolic
control properties.

Classes:
    LinLogBase: Abstract base class providing the lin-log steady-state equations.
    LinLogSymbolic2x2: Specialized solver for 2x2 full-rank systems using symbolic inversion.
    LinLogLinkMatrix: Solver using the link matrix decomposition for dimensionality reduction.
    LinLogLeastNorm: Solver using least-norm (pseudoinverse) solutions via LAPACK routines.
    LinLogTikhonov: Solver with Tikhonov regularization for ill-conditioned systems.
    LinLogPinv: Extended pseudoinverse solver supporting solution basis constraints.

References:
    - Smallbone et al. (2007). "Something from nothing − bridging the gap between
      constraint-based and kinetic modelling."
    - Visser & Heijnen (2003). "Dynamic simulation and metabolic re-design of a
      branched pathway using linlog kinetics."
    - St. John et al. (2019). "Bayesian inference of metabolic kinetics from genome-scale
      multiomics data." PLOS Computational Biology.
"""

import warnings

import numpy as np
import scipy as sp

import pytensor
import pytensor.tensor as at
import pytensor.tensor.slinalg

from bayesmca.pytensor_utils import RegularizedSolve, LeastSquaresSolve, lstsq_wrapper
from bayesmca.util import compute_smallbone_reduction, compute_waldherr_reduction


_floatX = pytensor.config.floatX


class LinLogBase:
    def __init__(self, N, Ex, Ey, v_star, reduction_method="smallbone"):
        """A class to perform the linear algebra underlying the
        decomposition method.


        Parameters
        ----------
        N : np.array
            The full stoichiometric matrix of the considered model. Must be of
            dimensions MxN
        Ex : np.array
            An NxM array of the elasticity coefficients for the given linlog
            model.
        Ey : np.array
            An NxP array of the elasticity coefficients for the external
            species.
        v_star : np.array
            A length M vector specifying the original steady-state flux
            solution of the model.
        lam : float
            The λ value to use for tikhonov regularization
        reduction_method : 'waldherr', 'smallbone', or None
            Type of stoichiometric decomposition to perform (default
            'smallbone')


        """
        self.nm, self.nr = N.shape
        self.ny = Ey.shape[1]

        self.N = N

        if reduction_method == "smallbone":
            self.Nr, self.L, _ = compute_smallbone_reduction(N, Ex, v_star)

        elif reduction_method == "waldherr":
            self.Nr, _, _ = compute_waldherr_reduction(N)

        elif reduction_method is None:
            self.Nr = N

        self.Ex = Ex
        self.Ey = Ey

        assert np.all(v_star >= 0), "reference fluxes should be nonnegative"
        if np.any(np.isclose(v_star, 0)):
            warnings.warn("v_star contains zero entries, this will cause problems")

        self.v_star = v_star

        assert Ex.shape == (self.nr, self.nm), "Ex is the wrong shape"
        assert Ey.shape == (self.nr, self.ny), "Ey is the wrong shape"
        assert len(v_star) == self.nr, "v_star is the wrong length"
        assert np.allclose(self.Nr @ v_star, 0), "reference not steady state"

    def _generate_default_inputs(self, Ex=None, Ey=None, en=None, yn=None):
        """Create matricies representing no perturbation is input is None."""
        if Ex is None:
            Ex = self.Ex

        if Ey is None:
            Ey = self.Ey

        if en is None:
            en = np.ones(self.nr)

        if yn is None:
            yn = np.zeros(self.ny)

        return Ex, Ey, en, yn

    def steady_state_mat(
        self,
        Ex=None,
        Ey=None,
        en=None,
        yn=None,
    ):
        """Calculate steady-state metabolite concentrations and fluxes using NumPy.

        Solves the lin-log steady-state equation:
            Nr @ diag(v* ⊙ e) @ (1 + Ex @ x + Ey @ y) = 0

        Rearranged to the linear system ``A @ x = b`` where:
            A = Nr @ diag(v* ⊙ e) @ Ex
            b = -Nr @ diag(v* ⊙ e) @ (1 + Ey @ y)

        Parameters
        ----------
        Ex : np.ndarray, optional
            (nr x nm) elasticity matrix for internal metabolites.
        Ey : np.ndarray, optional
            (nr x ny) elasticity matrix for external metabolites.
        en : np.ndarray, optional
            Length-nr vector of normalized enzyme activities (1.0 = reference).
        yn : np.ndarray, optional
            Length-ny vector of log-transformed external metabolite perturbations.

        Returns
        -------
        xn : np.ndarray
            Steady-state log-concentration perturbations (length nm).
        vn : np.ndarray
            Steady-state normalized flux vector (length nr).
        """
        Ex, Ey, en, yn = self._generate_default_inputs(Ex, Ey, en, yn)

        # Calculate steady-state concentrations using linear solve.
        N_hat = self.Nr @ np.diag(self.v_star * en)
        A = N_hat @ Ex
        b = -N_hat @ (np.ones(self.nr) + Ey @ yn)
        xn = self.solve(A, b)

        # Plug concentrations into the flux equation.
        vn = en * (np.ones(self.nr) + Ex @ xn + Ey @ yn)

        return xn, vn

    def steady_state_pytensor(self, Ex, Ey=None, en=None, yn=None, method="scan"):
        """Calculate steady-state concentrations and fluxes using PyTensor (symbolic).

        Equivalent to ``steady_state_mat`` but operates on PyTensor tensors,
        enabling automatic differentiation for use in PyMC probabilistic models.
        Supports batched computation over multiple experimental conditions.

        Parameters
        ----------
        Ex : pytensor.tensor
            (nr x nm) elasticity matrix (symbolic).
        Ey : pytensor.tensor, optional
            (nr x ny) external elasticity matrix.
        en : pytensor.tensor or np.ndarray
            (n_exp x nr) matrix of enzyme activities per experiment.
        yn : pytensor.tensor or np.ndarray
            (n_exp x ny) matrix of external metabolite perturbations.
        method : str
            'scan' uses pytensor.scan for loop; otherwise unrolls the loop.

        Returns
        -------
        xn : pytensor.tensor
            (n_exp x nm) steady-state concentration perturbations.
        vn : pytensor.tensor
            (n_exp x nr) steady-state normalized fluxes.
        """

        if Ey is None:
            Ey = at.as_tensor_variable(Ey)

        if isinstance(en, np.ndarray):
            en = np.atleast_2d(en)
            n_exp = en.shape[0]
        else:
            n_exp = en.eval().shape[0]

        if isinstance(yn, np.ndarray):
            yn = np.atleast_2d(yn)

        en = at.as_tensor_variable(en)
        yn = at.as_tensor_variable(yn)

        e_diag = en.dimshuffle(0, 1, "x") * np.diag(self.v_star)
        N_rep = self.Nr.reshape((-1, *self.Nr.shape)).repeat(n_exp, axis=0)
        N_hat = at.batched_dot(N_rep, e_diag)

        inner_v = Ey.dot(yn.T).T + np.ones(self.nr, dtype=_floatX)
        As = at.dot(N_hat, Ex)

        bs = at.batched_dot(-N_hat, inner_v.dimshuffle(0, 1, "x"))
        if method == "scan":
            xn, _ = pytensor.scan(
                lambda A, b: self.solve_pytensor(A, b), sequences=[As, bs], strict=True
            )
        else:
            xn_list = [None] * n_exp
            for i in range(n_exp):
                xn_list[i] = self.solve_pytensor(As[i], bs[i])
            xn = at.stack(xn_list)

        vn = en * (np.ones(self.nr) + at.dot(Ex, xn.T).T + at.dot(Ey, yn.T).T)

        return xn, vn

    def metabolite_control_coefficient(self, Ex=None, Ey=None, en=None, yn=None):
        """Calculate the metabolite control coefficient (MCC) matrix.

        The MCC matrix C^x has entries C^x_{i,j} = (∂ ln x_i / ∂ ln e_j),
        quantifying how a fractional change in enzyme j activity affects
        metabolite i concentration at steady state.

        Parameters
        ----------
        Ex, Ey, en, yn : optional
            See ``steady_state_mat`` for parameter descriptions.

        Returns
        -------
        Cx : np.ndarray
            (nm x nr) metabolite control coefficient matrix.
        """

        Ex, Ey, en, yn = self._generate_default_inputs(Ex, Ey, en, yn)

        xn, vn = self.steady_state_mat(Ex, Ey, en, yn)

        # Calculate the elasticity matrix at the new steady-state
        Ex_ss = np.diag(en / vn) @ Ex

        Cx = -self.solve(
            self.Nr @ np.diag(vn * self.v_star) @ Ex_ss,
            self.Nr @ np.diag(vn * self.v_star),
        )

        return Cx

    def flux_control_coefficient(self, Ex=None, Ey=None, en=None, yn=None):
        """Calculate the flux control coefficient (FCC) matrix.

        The FCC matrix C^v has entries C^v_{i,j} = (∂ ln v_i / ∂ ln e_j),
        quantifying how a fractional change in enzyme j activity affects
        flux i at steady state.  Related to the MCC via: C^v = I + Ex_ss @ C^x.

        Parameters
        ----------
        Ex, Ey, en, yn : optional
            See ``steady_state_mat`` for parameter descriptions.

        Returns
        -------
        Cv : np.ndarray
            (nr x nr) flux control coefficient matrix.
        """

        Ex, Ey, en, yn = self._generate_default_inputs(Ex, Ey, en, yn)

        xn, vn = self.steady_state_mat(Ex, Ey, en, yn)

        # Calculate the elasticity matrix at the new steady-state
        Ex_ss = np.diag(en / vn) @ Ex

        Cx = self.metabolite_control_coefficient(Ex, Ey, en, yn)
        Cv = np.eye(self.nr) + Ex_ss @ Cx

        return Cv


class LinLogSymbolic2x2(LinLogBase):
    """Lin-log solver for the special case of a 2x2 full-rank system.

    Uses closed-form symbolic matrix inversion (Cramer's rule) instead of
    a numerical solver. Only applicable when the reduced stoichiometric
    matrix yields a 2x2 linear system.
    """

    def solve(self, A, bi):
        a = A[0, 0]
        b = A[0, 1]
        c = A[1, 0]
        d = A[1, 1]

        A_inv = np.array([[d, -b], [-c, a]]) / (a * d - b * c)
        return A_inv @ bi

    def solve_pytensor(self, A, bi):
        a = A[0, 0]
        b = A[0, 1]
        c = A[1, 0]
        d = A[1, 1]

        A_inv = at.stacklists([[d, -b], [-c, a]]) / (a * d - b * c)
        return at.dot(A_inv, bi).squeeze()


class LinLogLinkMatrix(LinLogBase):
    """Lin-log solver using the link matrix for dimensionality reduction.

    Reduces the system via the link matrix L (from Smallbone decomposition),
    transforming the potentially rank-deficient system into a full-rank one:
        A_linked = A @ L,  then solve A_linked @ z = b,  x = L @ z.
    """
    def solve(self, A, b):
        A_linked = A @ self.L
        z = sp.linalg.solve(A_linked, b)
        return self.L @ z

    def solve_pytensor(self, A, b):
        A_linked = at.dot(A, self.L)
        z = pytensor.tensor.slinalg.solve(A_linked, b).squeeze()
        return at.dot(self.L, z)


class LinLogLeastNorm(LinLogBase):
    """Lin-log solver using least-norm (pseudoinverse) solutions.

    Solves ``Ax = b`` via LAPACK least-squares routines (e.g., GELSY, GELSD),
    returning the minimum-norm solution when the system is underdetermined.
    """

    def __init__(self, N, Ex, Ey, v_star, driver="gelsy", **kwargs):
        self.driver = driver
        LinLogBase.__init__(self, N, Ex, Ey, v_star, **kwargs)

    def solve(self, A, b):
        return lstsq_wrapper(A, b, self.driver)

    def solve_pytensor(self, A, b):
        rsolve_op = LeastSquaresSolve(driver=self.driver, b_ndim=2)
        return rsolve_op(A, b).squeeze()


class LinLogTikhonov(LinLogBase):
    """Lin-log solver with Tikhonov regularization.

    Solves ``min ||Ax - b||² + λ||x||²`` via the normal equations with a
    regularization parameter λ. Useful for ill-conditioned systems where the
    unregularized solve is numerically unstable.
    """

    def __init__(self, N, Ex, Ey, v_star, lambda_=None, **kwargs):
        self.lambda_ = lambda_ if lambda_ else 0
        assert self.lambda_ >= 0, "lambda must be positive"

        LinLogBase.__init__(self, N, Ex, Ey, v_star, **kwargs)

    def solve(self, A, b):
        A_hat = A.T @ A + self.lambda_ * np.eye(A.shape[1])
        b_hat = A.T @ b

        cho = sp.linalg.cho_factor(A_hat)
        return sp.linalg.cho_solve(cho, b_hat)

    def solve_pytensor(self, A, b):
        rsolve_op = RegularizedSolve(self.lambda_)
        return rsolve_op(A, b).squeeze()


class LinLogPinv(LinLogLeastNorm):
    """Lin-log solver using pseudoinverse with a solution basis constraint.

    Extends the least-norm solver by projecting the solution onto a specified
    basis, useful when additional constraints on the metabolite concentration
    space are available from prior knowledge or sampling.
    """
    def steady_state_pytensor(
        self,
        Ex,
        Ey=None,
        en=None,
        yn=None,
        solution_basis=None,
        method="scan",
        driver="gelsy",
    ):
        """Calculate a the steady-state transformed metabolite concentrations
        and fluxes using pytensor.

        Ex, Ey, en and yn should be pytensor matrices

        solution_basis is a (n_exp, nr) pytensor matrix of the current solution
        basis.

        solver: function
            A function to solve Ax = b for a (possibly) singular A. Should
            accept pytensor matrices A and b, and return a symbolic x.
        """

        if Ey is None:
            Ey = at.as_tensor_variable(Ey)

        if isinstance(en, np.ndarray):
            en = np.atleast_2d(en)
            n_exp = en.shape[0]
        else:
            n_exp = en.tag.test_value.shape[0]

        if isinstance(yn, np.ndarray):
            yn = np.atleast_2d(yn)

        en = at.as_tensor_variable(en)
        yn = at.as_tensor_variable(yn)

        e_diag = en.dimshuffle(0, 1, "x") * np.diag(self.v_star)
        N_rep = self.Nr.reshape((-1, *self.Nr.shape)).repeat(n_exp, axis=0)
        N_hat = at.batched_dot(N_rep, e_diag)

        inner_v = Ey.dot(yn.T).T + np.ones(self.nr, dtype=_floatX)
        As = at.dot(N_hat, Ex)

        bs = at.batched_dot(-N_hat, inner_v.dimshuffle(0, 1, "x"))

        # Here we have to redefine the entire function, since we have to pass
        # an additional argument to solve.
        def pinv_solution(A, b, basis=None):
            A_pinv = at.nlinalg.pinv(A)
            x_ln = at.dot(A_pinv, b).squeeze()
            x = x_ln + at.dot((at.eye(self.nm) - at.dot(A_pinv, A)), basis)
            return x

        if method == "scan":
            xn, _ = pytensor.scan(
                lambda A, b, w: pinv_solution(A, b, basis=w),
                sequences=[As, bs, solution_basis],
                strict=True,
            )

        else:
            xn_list = [None] * n_exp
            for i in range(n_exp):
                xn_list[i] = pinv_solution(As[i], bs[i], solution_basis[i])
            xn = at.stack(xn_list)

        vn = en * (np.ones(self.nr) + at.dot(Ex, xn.T).T + at.dot(Ey, yn.T).T)

        return xn, vn
