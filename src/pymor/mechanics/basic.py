# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config
from scipy.linalg import solve, solve_triangular


if config.HAVE_FENICS:
    import inspect
    import dolfin as df
    import ufl
    import numpy as np
    from fenics_modules.helpers import local_project

    from pymor.bindings.fenics import FenicsVector, FenicsVectorSpace, FenicsMatrixOperator
    from pymor.core.defaults import defaults
    from pymor.core.interfaces import BasicInterface
    from pymor.operators.basic import OperatorBase
    from pymor.operators.ei import EmpiricalInterpolatedOperator
    from pymor.operators.numpy import NumpyMatrixOperator
    from pymor.vectorarrays.interfaces import _create_random_values, VectorArrayInterface
    from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorSpace
    from pymor.vectorarrays.numpy import NumpyVectorSpace

    class MechanicsOperator(OperatorBase):
        r"""Wraps the linearized principle of virtual power (given as FEniCS forms) as an |Operator|.

        .. math::
            \int_{\Omega} \varepsilon(v) \cdot \frac{\partial \sigma}{\partial \varepsilon}(u^n;\mu)
            \cdot \varepsilon(u^{n+1}) \; \mathrm{d}x = f_{\mathrm{ext}} - \int_{\Omega} \sigma(u^n;\mu)
            \cdot \varepsilon(v) \; \mathrm{d}x\, \quad \forall v \in \mathbb{V}.

        """

        linear = True

        @defaults('restriction_method')
        def __init__(self, jacobian_form, operator_form, source_space, range_space,
                     material=None, dirichlet_bc=None, subdomain_data=None,
                     parameter_setter=None, parameter_type=None, solver_options=None,
                     restriction_method='assemble_local', name=None):
            assert restriction_method in ('assemble_local', 'submesh')
            assert material is None or hasattr(material, 'update_history')
            assert isinstance(dirichlet_bc, list) or dirichlet_bc == None
            if subdomain_data:
                # FIXME: subdomain data is not needed. but assemble_local should be forced in these
                # cases
                assert restriction_method == 'assemble_local'# FIXME: submesh will not work
            self.jacobian = jacobian_form
            self.operator = operator_form
            self.source = source_space
            self.range = range_space
            self.material = material
            self.dirichlet_bc = dirichlet_bc
            self.parameter_setter = parameter_setter
            self.build_parameter_type(parameter_type)
            self.solver_options = solver_options
            self.restriction_method = restriction_method
            self.name = name

        def _set_mu(self, mu=None):
            mu = self.parse_parameter(mu)
            if self.parameter_setter:
                self.parameter_setter(mu)

        def apply(self, U, mu=None):
            assert U in self.source
            self._set_mu(mu)
            R = []
            for u in U._list:
                if self.material:
                    self.material.update_history(u.impl, self.range)
                r = df.assemble(self.operator)
                if self.dirichlet_bc:
                    for bc in self.dirichlet_bc:
                        bc.apply(r)
                R.append(r)
            return self.range.make_array(R)

        def jacobian(self, U, mu=None):
            assert U in self.source and len(U) == 1
            self._set_mu(mu)
            if self.material:
                self.material.update_history(U._list[0].impl, self.range)
            matrix = df.assemble(self.jacobian)
            if self.dirichlet_bc:
                for bc in self.dirichlet_bc:
                    bc.apply(matrix)
            return FenicsMatrixOperator(matrix, self.source.V, self.range.V)

        def restricted(self, dofs):
            assert self.source.V.mesh().id() == self.range.V.mesh().id()

            # first determine affected cells
            self.logger.info('Computing affected cells ...')
            mesh = self.source.V.mesh()
            range_dofmap = self.range.V.dofmap()
            affected_cell_indices = set()
            for c in df.cells(mesh):
                cell_index = c.index()
                local_dofs = range_dofmap.cell_dofs(cell_index)
                for ld in local_dofs:
                    if ld in dofs:
                        affected_cell_indices.add(cell_index)
                        continue
            affected_cell_indices = list(sorted(affected_cell_indices))
            affected_cells = [df.Cell(mesh, ci) for ci in affected_cell_indices]

            # increase stencil if needed
            # TODO

            # determine source dofs
            self.logger.info('Computing source DOFs ...')
            source_dofmap = self.source.V.dofmap()
            source_dofs = set()
            for cell_index in affected_cell_indices:
                local_dofs = source_dofmap.cell_dofs(cell_index)
                source_dofs.update(local_dofs)
            source_dofs = np.array(sorted(source_dofs), dtype=np.intc)

            if self.restriction_method == 'assemble_local':
                # range local-to-restricted dof mapping
                to_restricted = np.zeros(self.range.dim, dtype=np.int32)
                to_restricted[:] = len(dofs)
                to_restricted[dofs] = np.arange(len(dofs))
                range_local_restricted = np.array([to_restricted[range_dofmap.cell_dofs(ci)]
                                                   for ci in affected_cell_indices])

                # source local-to-restricted dof mapping
                to_restricted = np.zeros(self.source.dim, dtype=np.int32)
                to_restricted[:] = len(source_dofs)
                to_restricted[source_dofs] = np.arange(len(source_dofs))
                source_local_restricted = np.array([to_restricted[source_dofmap.cell_dofs(ci)]
                                                   for ci in affected_cell_indices])

                # compute dirichlet DOFs
                if self.dirichlet_bc:
                    self.logger.warn('Dirichlet DOF handling will only work for constant, non-paramentric '
                                     'Dirichlet boundary conditions')
                    for bc in self.dirichlet_bc:
                        v1 = self.source.zeros()._list[0].impl
                        v1[:] = 42
                        v2 = self.source.zeros()._list[0].impl
                        v2[:] = 0
                        bc.apply(v1)
                        bc.apply(v2)
                    dir_dofs = [i for i in range(self.source.dim) if (v1[i] != 42) or (v2[i] != 0)]
                    # determine whether dir_dofs are in interpolation dofs
                    intersection = set(dofs.tolist()).intersection(set(dir_dofs))
                    if len(intersection) < 1:
                        dir_dofs_r = None
                        dir_vals_r = None
                        dir_dofs_r_source = None
                    else:
                        dir_dofs_r, dir_vals_r = zip(*((i, v1[dof]) for i, dof in enumerate(dofs) if dof in dir_dofs))
                        dir_dofs_r = np.array(dir_dofs_r, dtype=np.int32)
                        dir_vals_r = np.array(dir_vals_r)
                        dir_dofs_r_source = to_restricted[dofs[dir_dofs_r]]
                else:
                    dir_dofs_r = None
                    dir_vals_r = None
                    dir_dofs_r_source = None
                return (
                    RestrictedMechanicsOperatorAssembleLocal(self, np.array(dofs), source_dofs.copy(), affected_cells,
                                                          source_local_restricted, range_local_restricted,
                                                          dir_dofs_r, dir_vals_r, dir_dofs_r_source),
                    source_dofs
                )

            elif self.restriction_method == 'submesh':
                raise NotImplementedError

    class RestrictedMechanicsOperatorAssembleLocal(OperatorBase):

        linear = False

        def __init__(self, operator, range_dofs, source_dofs, cells, source_local_restricted, range_local_restricted,
                     dirichlet_dofs, dirichlet_values, dirichlet_source_dofs):
            self.source = NumpyVectorSpace(len(source_dofs))
            self.range = NumpyVectorSpace(len(range_dofs))
            self.operator = operator
            self.range_dofs = range_dofs
            self.source_dofs = source_dofs
            self.cells = cells
            self.source_local_restricted = source_local_restricted
            self.range_local_restricted = range_local_restricted
            self.dirichlet_dofs = dirichlet_dofs
            self.dirichlet_values = dirichlet_values
            self.dirichlet_source_dofs = dirichlet_source_dofs
            self.build_parameter_type(operator)

        def apply(self, U, mu=None):
            assert U in self.source
            operator = self.operator
            operator._set_mu(mu)
            R = np.zeros((len(U), self.range.dim + 1))
            for u, r in zip(U.data, R):
                if self.operator.material:
                    self.operator.material.update_history(u, self.operator.range,
                                                          source_dofs=self.source_dofs)
                for cell, local_restricted in zip(self.cells, self.range_local_restricted):
                    local_evaluations = df.assemble_local(operator.operator, cell)
                    r[local_restricted] += local_evaluations
                if self.dirichlet_values:
                    r[self.dirichlet_dofs] = u[self.dirichlet_source_dofs] - self.dirichlet_values
            return self.range.make_array(R[:, :-1])

        def jacobian(self, U, mu=None):
            assert U in self.source and len(U) == 1
            operator = self.operator
            source_vec = operator.source_function.vector()# FIXME: there is no source function
            operator._set_mu(mu)
            J = np.zeros((self.range.dim + 1, self.source.dim + 1))
            source_vec[self.source_dofs] = U.data[0]
            for cell, range_local_restricted, source_local_restricted in zip(self.cells,
                                                                             self.range_local_restricted,
                                                                             self.source_local_restricted):
                # history update is already done when operator.apply(U, mu) is called
                # in lines 93,126 of newton.py
                local_matrix = df.assemble_local(operator.jacobian, cell)
                J[tuple(np.meshgrid(range_local_restricted, source_local_restricted, indexing='ij'))] += local_matrix
            if self.dirichlet_dofs is not None:
                J[self.dirichlet_dofs, :] = 0.# if self.dirichlet_dofs is None, don't do this. this will
                #  set all elements of J to zero
                #  J[np.meshgrid(self.dirichlet_dofs, self.dirichlet_source_dofs, indexing='ij')] = 1.
                J[tuple(np.meshgrid(self.dirichlet_dofs, self.dirichlet_source_dofs, indexing='ij'))] = 1.
            return NumpyMatrixOperator(J[:-1, :-1])

        def restricted(self, dofs):
            raise NotImplementedError

    class MechanicsEmpiricalInterpolatedOperator(EmpiricalInterpolatedOperator):
        """Interpolate an 'MechanicsOperator' using Empirical Operator Interpolation."""

        def __init__(self, operator, interpolation_dofs, collateral_basis, triangular,
                     assembled_basis=None, solver_options=None, name=None):
            if assembled_basis:
                assert isinstance(assembled_basis, VectorArrayInterface)
                self.assembled_basis = assembled_basis.copy()
                triangular = False# interpolation matrix will not be triangular for UDEIM
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

