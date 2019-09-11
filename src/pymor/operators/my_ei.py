# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import dolfin as df
from scipy.linalg import solve, solve_triangular


from pymor.bindings.fenics import FenicsVectorSpace, FenicsVector
from pymor.operators.basic import OperatorBase
from pymor.operators.constructions import VectorArrayOperator, Concatenation, ComponentProjection, ZeroOperator
from pymor.operators.interfaces import OperatorInterface
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.vectorarrays.numpy import NumpyVectorSpace


class FenicsEmpiricalInterpolatedOperator(OperatorBase):
    """Interpolate an |Operator| using Empirical Operator Interpolation.

    Let `L` be an |Operator|, `0 <= c_1, ..., c_M < L.range.dim` indices
    of interpolation DOFs and let `b_1, ..., b_M in R^(L.range.dim)` be collateral
    basis vectors. If moreover `ψ_j(U)` denotes the j-th component of `U`, the
    empirical interpolation `L_EI` of `L` w.r.t. the given data is given by ::

      |                M
      |   L_EI(U, μ) = ∑ b_i⋅λ_i     such that
      |               i=1
      |
      |   ψ_(c_i)(L_EI(U, μ)) = ψ_(c_i)(L(U, μ))   for i=0,...,M

    Since the original operator only has to be evaluated at the given interpolation
    DOFs, |EmpiricalInterpolatedOperator| calls
    :meth:`~pymor.operators.interfaces.OperatorInterface.restricted`
    to obtain a restricted version of the operator which is used
    to quickly obtain the required evaluations. If the `restricted` method, is not
    implemented, the full operator will be evaluated (which will lead to
    the same result, but without any speedup).

    The interpolation DOFs and the collateral basis can be generated using
    the algorithms provided in the :mod:`pymor.algorithms.ei` module.


    Parameters
    ----------
    operator
        The |Operator| to interpolate.
    interpolation_dofs
        List or 1D |NumPy array| of the interpolation DOFs `c_1, ..., c_M`.
    collateral_basis
        |VectorArray| containing the collateral basis `b_1, ..., b_M`.
    triangular
        If `True`, assume that ψ_(c_i)(b_j) = 0  for i < j, which means
        that the interpolation matrix is triangular.
    solver_options
        The |solver_options| for the operator.
    name
        Name of the operator.
    """

    def __init__(self, operator, interpolation_dofs, collateral_basis, triangular,
                 solver_options=None, name=None, estimate_error=None):
        assert isinstance(operator, OperatorInterface)
        assert isinstance(collateral_basis, VectorArrayInterface)
        if operator.restriction_method == 'IQP':
            assert collateral_basis in operator.quadrature_range
        else:
            assert collateral_basis in operator.range
        assert len(interpolation_dofs) == len(collateral_basis)

        self.build_parameter_type(operator)
        self.source = operator.source
        if estimate_error:
            self.range = operator.quadrature_range
            self.name = name or f'{operator.name}_interpolated_error_estimator'
        else:
            self.range = operator.range
            self.name = name or f'{operator.name}_interpolated'
        self.linear = operator.linear
        self.solver_options = solver_options
        interpolation_dofs = np.array(interpolation_dofs, dtype=np.int32)
        self.interpolation_dofs = interpolation_dofs
        self.triangular = triangular

        if len(interpolation_dofs) > 0:
            try:
                self.restricted_operator, self.source_dofs = operator.restricted(interpolation_dofs)
            except NotImplementedError:
                self.logger.warning('Operator has no "restricted" method. The full operator will be evaluated.')
                self.operator = operator
            interpolation_matrix = collateral_basis.dofs(interpolation_dofs).T
            self.interpolation_matrix = interpolation_matrix
            self.collateral_basis = collateral_basis.copy()
            if operator.restriction_method == 'IQP':
                self.logger.info('Assembling deim basis functions for restriction method ' +
                                 f'{operator.restriction_method} ...')
                Q = operator.quadrature_range.V
                QUAD_DEG = Q.ufl_element().degree()
                metadata = {"quadrature_degree": QUAD_DEG, "quadrature_scheme": "default"}
                dx = df.dx(metadata=metadata)
                v = df.TestFunction(operator.range.V)
                W = collateral_basis.data.T
                n_idofs = W.shape[1]
                W_a = operator.range.empty()
                tmp = df.Function(Q, name='basis')
                for i in range(n_idofs):
                    tmp.vector()[:] = W[:, i]
                    W_a.append(operator.range.make_array([
                        df.assemble(v * tmp * dx)
                    ]))
                assert W_a.data.shape == (n_idofs, operator.range.dim)
                # TODO: check orthonormality of assembled basis W_a
                self.assembled_basis = W_a.copy()
                self.logger.info('Precomputing MDEIM for restriction method ' +
                                f'{operator.restriction_method} ...')
                self.MDEIM = np.dot(W_a.data.T, np.linalg.inv(interpolation_matrix))
                if self.name.split('_', 2)[-1] == 'error_estimator':
                    # unassembled version used in error estimation
                    self.MDEIM = np.dot(W, np.linalg.inv(interpolation_matrix))
            else:
                self.collateral_basis = collateral_basis.copy()
                self.MDEIM = None

    def apply(self, U, mu=None):
        mu = self.parse_parameter(mu)
        if len(self.interpolation_dofs) == 0:
            return self.range.zeros(len(U))

        if hasattr(self, 'restricted_operator'):
            U_dofs = NumpyVectorSpace.make_array(U.dofs(self.source_dofs))
            AU = self.restricted_operator.apply(U_dofs, mu=mu)
        else:
            AU = NumpyVectorSpace.make_array(self.operator.apply(U, mu=mu).dofs(self.interpolation_dofs))
        if self.MDEIM is not None:
            V = self.range.V
            tmp = df.Function(V)
            tmp.vector()[:] = np.dot(self.MDEIM, AU.to_numpy().reshape(AU.dim,))
            # TODO: is this more costly than solving for the coefficients in every call to apply?
            return self.range.make_array([tmp.vector()])
        else:
            try:
                if self.triangular:
                    interpolation_coefficients = solve_triangular(self.interpolation_matrix, AU.to_numpy().T,
                                                                  lower=True, unit_diagonal=True).T
                else:
                    interpolation_coefficients = solve(self.interpolation_matrix, AU.to_numpy().T).T
            except ValueError:  # this exception occurs when AU contains NaNs ...
                interpolation_coefficients = np.empty((len(AU), len(self.collateral_basis))) + np.nan
            return self.collateral_basis.lincomb(interpolation_coefficients)

    def jacobian(self, U, mu=None):
        mu = self.parse_parameter(mu)
        options = self.solver_options.get('jacobian') if self.solver_options else None

        if len(self.interpolation_dofs) == 0:
            if isinstance(self.source, NumpyVectorSpace) and isinstance(self.range, NumpyVectorSpace):
                return NumpyMatrixOperator(np.zeros((self.range.dim, self.source.dim)), solver_options=options,
                                           source_id=self.source.id, range_id=self.range.id,
                                           name=self.name + '_jacobian')
            else:
                return ZeroOperator(self.range, self.source, name=self.name + '_jacobian')
        elif hasattr(self, 'operator'):
            return EmpiricalInterpolatedOperator(self.operator.jacobian(U, mu=mu), self.interpolation_dofs,
                                                 self.collateral_basis, self.triangular,
                                                 solver_options=options, name=self.name + '_jacobian')
        else:
            restricted_source = self.restricted_operator.source
            U_dofs = restricted_source.make_array(U.dofs(self.source_dofs))
            JU = self.restricted_operator.jacobian(U_dofs, mu=mu) \
                                         .apply(restricted_source.make_array(np.eye(len(self.source_dofs))))
            try:
                if self.triangular:
                    interpolation_coefficients = solve_triangular(self.interpolation_matrix, JU.to_numpy().T,
                                                                  lower=True, unit_diagonal=True).T
                else:
                    interpolation_coefficients = solve(self.interpolation_matrix, JU.to_numpy().T).T
            except ValueError:  # this exception occurs when AU contains NaNs ...
                interpolation_coefficients = np.empty((len(JU), len(self.collateral_basis))) + np.nan
            J = self.collateral_basis.lincomb(interpolation_coefficients)
            if isinstance(J.space, NumpyVectorSpace):
                J = NumpyMatrixOperator(J.to_numpy().T, range_id=self.range.id)
            else:
                J = VectorArrayOperator(J)
            return Concatenation([J, ComponentProjection(self.source_dofs, self.source)],
                                 solver_options=options, name=self.name + '_jacobian')
