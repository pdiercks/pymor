# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np
from scipy.linalg import solve, solve_triangular

from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.vectorarrays.interfaces import VectorArrayInterface
from pymor.vectorarrays.numpy import NumpyVectorSpace


class MechanicsEmpiricalInterpolatedOperator(EmpiricalInterpolatedOperator):
    """Interpolate a 'MechanicsOperator' using Empirical Operator Interpolation."""

    def __init__(self, operator, interpolation_dofs, collateral_basis, triangular,
                 assembled_basis=None, solver_options=None, name=None):
        if assembled_basis:
            assert isinstance(assembled_basis, VectorArrayInterface)
            self.assembled_basis = assembled_basis.copy()
            triangular = False  # interpolation matrix will not be triangular for UDEIM
        super().__init__(operator, interpolation_dofs, collateral_basis, triangular,
                         solver_options=solver_options, name=name)

    def apply(self, U, mu=None):
        mu = self.parse_parameter(mu)
        if len(self.interpolation_dofs) == 0:
            return self.range.zeros(len(U))

        if hasattr(self, 'restricted_operator'):
            # assert U in self.operator.op.source # should be DG space
            # restricted_operator.source.random()
            U_dofs = NumpyVectorSpace.make_array(U.dofs(self.source_dofs))
            AU = self.restricted_operator.apply(U_dofs, mu=mu)
        else:
            AU = NumpyVectorSpace.make_array(self.operator.apply(U, mu=mu).dofs(self.interpolation_dofs))
        try:
            if self.triangular:
                interpolation_coefficients = solve_triangular(self.interpolation_matrix, AU.to_numpy().T,
                                                              lower=True, unit_diagonal=True).T
            else:
                interpolation_coefficients = solve(self.interpolation_matrix, AU.to_numpy().T).T
        except ValueError:  # this exception occurs when AU contains NaNs ...
            interpolation_coefficients = np.empty((len(AU), len(self.collateral_basis))) + np.nan

        if self.assembled_basis:
            return self.assembled_basis.lincomb(interpolation_coefficients)
        else:
            return self.collateral_basis.lincomb(interpolation_coefficients)
