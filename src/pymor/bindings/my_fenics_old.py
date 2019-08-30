# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_FENICS:
    import types
    import inspect
    import ufl
    import dolfin as df
    import numpy as np
    import copy


    from pymor.bindings.fenics import FenicsVectorSpace, FenicsVector
    from pymor.core.defaults import defaults
    from pymor.core.interfaces import BasicInterface
    from pymor.operators.basic import OperatorBase
    from pymor.operators.numpy import NumpyMatrixOperator
    from pymor.vectorarrays.interfaces import _create_random_values
    from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorSpace
    from pymor.vectorarrays.numpy import NumpyVectorSpace

    class FenicsMatrixOperator(OperatorBase):
        """Wraps a FEniCS matrix as an |Operator|."""

        linear = True

        def __init__(self, matrix, source_space, range_space, solver_options=None, name=None):
            assert matrix.rank() == 2
            self.source = FenicsVectorSpace(source_space)
            self.range = FenicsVectorSpace(range_space)
            self.matrix = matrix
            self.solver_options = solver_options
            self.name = name

        def apply(self, U, mu=None):
            assert U in self.source
            R = self.range.zeros(len(U))
            for u, r in zip(U._list, R._list):
                self.matrix.mult(u.impl, r.impl)
            return R

        def apply_adjoint(self, V, mu=None):
            assert V in self.range
            U = self.source.zeros(len(V))
            for v, u in zip(V._list, U._list):
                self.matrix.transpmult(v.impl, u.impl)  # there are no complex numbers in FEniCS
            return U

        def apply_inverse(self, V, mu=None, least_squares=False):
            #  assert V in self.range# TODO: find out why this fails for
            #  FenicsNonaffineVectorOperator_interpolated
            # need to pass the same object, not just an equivalent space???
            if least_squares:
                raise NotImplementedError
            R = self.source.zeros(len(V))
            options = self.solver_options.get('inverse') if self.solver_options else None
            for r, v in zip(R._list, V._list):
                _apply_inverse(self.matrix, r.impl, v.impl, options)
            return R

        def _assemble_lincomb(self, operators, coefficients, identity_shift=0., solver_options=None, name=None):
            if not all(isinstance(op, FenicsMatrixOperator) for op in operators):
                return None
            if identity_shift != 0:
                return None
            assert not solver_options

            if coefficients[0] == 1:
                matrix = operators[0].matrix.copy()
            else:
                matrix = operators[0].matrix * coefficients[0]
            for op, c in zip(operators[1:], coefficients[1:]):
                matrix.axpy(c, op.matrix, False)
                # in general, we cannot assume the same nonzero pattern for # all matrices. how to improve this?

            return FenicsMatrixOperator(matrix, self.source.V, self.range.V, name=name)

    class FenicsNonaffineOperator(OperatorBase):
        """Wraps a nonaffine linear form as an |Operator| mapping
        vectors to vectors via the apply method.

        Add doc
        """

        linear = True# linear in the sense that it is a linear form f(v ; u^n) not depending on
        #  u^{n+1}

        @defaults('restriction_method')
        def __init__(self, source_space, range_space, material, form, dirichlet_bc=None,
                     parameter_setter=None, parameter_type=None, solver_options=None,
                     restriction_method='submesh', name=None, subdomain_data=None):
            assert restriction_method in ('submesh', 'iqp')# TODO: add iqp variant
            assert inspect.isclass(type(material))
            assert isinstance(form, ufl.form.Form)
            assert isinstance(dirichlet_bc, list) or dirichlet_bc == None
            self.source = source_space
            self.range = range_space
            self.material = material
            self.form = form
            self.dirichlet_bc = dirichlet_bc
            self.parameter_setter = parameter_setter
            self.build_parameter_type(parameter_type)
            self.solver_options = solver_options
            self.name = name
            self.subdomains = subdomain_data

        def _set_mu(self, mu=None):
            mu = self.parse_parameter(mu)
            if self.parameter_setter:
                self.parameter_setter(mu)

        def apply(self, U, mu=None):
            assert U in self.source
            self._set_mu(mu)
            R = []
            #  source_vec = self.source_function.vector()
            for u in U._list:
                #  source_vec[:] = u.impl
                self.material.update_history(u.impl, self.range)
                r = df.assemble(self.form)
                if self.dirichlet_bc:
                    for bc in self.dirichlet_bc:
                        bc.apply(r)
                R.append(r)
            return self.range.make_array(R)

        def restricted(self, dofs):
            assert self.source.V.mesh().id() == self.range.V.mesh().id()
            # TODO: if statement for deim variants, i.e. restriction_method to be used

            # first determine affected cells
            self.logger.info('Computing affected cells ...')
            mesh = self.range.V.mesh()
            # TODO: get subdomain data as well, affected cells should be in matrix domain only
            range_dofmap = self.range.V.dofmap()
            affected_cell_indices = set()
            if self.subdomains:
                # loop over cells in subdomain marked with 1 only
                cell_iterator = df.SubsetIterator(self.subdomains, 1)# matrix domain
                # TODO: how to avoid hardcoding this?
            else:
                cell_iterator = df.cells(mesh)
            for c in cell_iterator:
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

            # generate restricted spaces
            self.logger.info("Building submesh ...")
            ei_domain = df.MeshFunction("size_t", mesh, mesh.geometry().dim())
            for ci in affected_cell_indices:
                ei_domain.set_value(ci, 1)
                # TODO: set value according to subdomain data
            ei_mesh = df.SubMesh(mesh, ei_domain, 1)
            # TODO: does this work for meshes with subdomains?
            # TODO: redefine measure dx ...
            # TODO: use only subdomain corresponding to viscoelastic material

            # build restricted form and material
            self.logger.info("Building ufl form on submesh ...")
            V_r_source = df.FunctionSpace(ei_mesh, self.source.V.ufl_element())
            V_r_range = df.FunctionSpace(ei_mesh, self.range.V.ufl_element())
            assert V_r_source.dim() == len(source_dofs)
            if self.source.V != self.range.V:
                assert all(arg.ufl_function_space() != self.source.V for arg in self.form.arguments())
            mat_r = copy.copy(self.material)# make real copy, deepcopy does not work due to dolfin
            #  define reduced quadrature spaces
            scalar_space_r_range = df.FunctionSpace(ei_mesh,
                                                    self.material._scalar_space.ufl_element())
            tensor_space_r_range = df.FunctionSpace(ei_mesh,
                                                    self.material._tensor_space.ufl_element())
            mat_r._scalar_space = scalar_space_r_range
            mat_r._tensor_space = tensor_space_r_range
            mat_r.initialize_history_data()
            # FIXME: this is specific to the nonaffine form to wrap ...
            history_r = {k: mat_r._history_data[k] for k in ("eps", "csi", "sigma")}

            args = tuple(
                (df.function.argument.Argument(V_r_range, arg.number(), arg.part())
                 if arg.ufl_function_space() == self.range.V else arg)
                for arg in self.form.arguments()
            )
            # form.subdomain_data()
            form_r = ufl.replace_integral_domains(
                self.form(*args, coefficients={self.material._history_data[k]: history_r[k] for k in history_r.keys()}),
                ei_mesh.ufl_domain()
            )
            if self.dirichlet_bc:
                bcs_r = []
                #  bcs = self.dirichlet_bc
                #  for bc in bcs:
                    #  if not bc.user_subdomain():
                    #      raise NotImplementedError
                    # TODO: bc.user_domain() is None when using MeshFunction for boundaries
                    #  bc_r = df.DirichletBC(V_r_range, bc.value(), bc.user_subdomain(), bc.method())
                    #  bcs_r.append(bc_r)
            else:
                bcs_r = None

            # source dof mapping
            self.logger.info('Computing source DOF mapping ...')
            u = df.Function(self.source.V)
            u_vec = u.vector()
            restricted_source_dofs = []
            for source_dof in source_dofs:
                u_vec.zero()
                u_vec[source_dof] = 1
                u_r = df.interpolate(u, V_r_source)
                u_r = u_r.vector().get_local()
                if not np.all(np.logical_or(np.abs(u_r) < 1e-10, np.abs(u_r - 1.) < 1e-10)):
                    raise NotImplementedError
                r_dof = np.where(np.abs(u_r - 1.) < 1e-10)[0]
                if not len(r_dof) == 1:
                    raise NotImplementedError
                restricted_source_dofs.append(r_dof[0])
            restricted_source_dofs = np.array(restricted_source_dofs, dtype=np.int32)
            assert len(set(restricted_source_dofs)) == len(source_dofs)

            # range dof mapping
            self.logger.info("Computing range dof mapping ...")
            u = df.Function(self.range.V)
            u_vec = u.vector()
            restricted_range_dofs = []
            for range_dof in dofs:# dofs = interpolation dofs from original mesh
                u_vec.zero()
                u_vec[range_dof] = 1
                u_r = df.interpolate(u, V_r_range)
                u_r = u_r.vector().get_local()
                if not np.all(np.logical_or(np.abs(u_r) < 1e-10, np.abs(u_r - 1.) < 1e-10)):
                    raise NotImplementedError
                r_dof = np.where(np.abs(u_r - 1.) < 1e-10)[0]
                if not len(r_dof) == 1:
                    raise NotImplementedError
                restricted_range_dofs.append(r_dof[0])
            restricted_range_dofs = np.array(restricted_range_dofs, dtype=np.int32)
            op_r = FenicsNonaffineOperator(FenicsVectorSpace(V_r_source),
                                           FenicsVectorSpace(V_r_range),
                                           material=mat_r,
                                           form=form_r,
                                           dirichlet_bc=bcs_r,
                                           parameter_setter=self.parameter_setter,
                                           parameter_type=self.parameter_type)

            return (RestrictedFenicsNonaffineOperatorSubMesh(op_r, restricted_range_dofs),
                    source_dofs[np.argsort(restricted_source_dofs)])

    class RestrictedFenicsNonaffineOperatorSubMesh(OperatorBase):
        """Restricted version of FenicsNonaffineOperator"""

        linear = True

        def __init__(self, operator, restricted_range_dofs):
            self.source = NumpyVectorSpace(operator.source.dim)
            self.range = NumpyVectorSpace(len(restricted_range_dofs))
            self.op = operator
            self.restricted_range_dofs = restricted_range_dofs
            self.build_parameter_type(operator)

        def apply(self, U, mu=None):# same as RestrictedFenicsOperatorSubMesh
            assert U in self.source
            UU = self.op.source.zeros(len(U))
            for uu, u in zip(UU._list, U.data):
                uu.impl[:] = u
            VV = self.op.apply(UU, mu=mu)# material update is done in op.apply
            V = self.range.zeros(len(VV))
            for v, vv in zip(V.data, VV._list):
                v[:] = vv.impl[self.restricted_range_dofs]
            return V

    def local_project(v, V, u=None):
        """project v onto V and store the values in u"""
        QUAD_DEG = V.ufl_element().degree()
        metadata = {"quadrature_degree": QUAD_DEG, "quadrature_scheme": "default"}
        dx = df.dx(metadata=metadata)
        v_trial = df.TrialFunction(V)
        v_test = df.TestFunction(V)
        a_proj = df.inner(v_trial, v_test) * dx
        b_proj = df.inner(v, v_test) * dx
        solver = df.LocalSolver(a_proj, b_proj)
        solver.factorize()
        if u is None:
            u = df.Function(V)
            solver.solve_local_rhs(u)
            return u
        else:
            solver.solve_local_rhs(u)
            return

    class FenicsNonaffineVectorOperator(OperatorBase):
        """Wraps a nonaffine linear form as a vector-like |Operator|.

        Given a vector `v` of dimension `d`, this class represents
        the operator ::
            op: R^1 ----> R^d
                x   |---> x·v(U, µ)
        """

        linear = True
        source = NumpyVectorSpace(1)

        @defaults('restriction_method')
        def __init__(self, form, range_space, quadrature_space=None, dirichlet_bc=None,
                     parameter_setter=None, parameter_type=None, solver_options=None,
                     restriction_method='submesh', numpy_form=None, name=None):

            if restriction_method == 'submesh':
                assert len(form.arguments()) == 1
            elif restriction_method == 'IQP':
                self.numpy_form = numpy_form
            else:
                raise NotImplementedError

            self.form = form
            self.range = range_space# should always be fe space
            self.quadrature_range = quadrature_space# Q
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
            # TODO: application to len(U) > 1?, see FenicsOperator
            assert U in self.source
            self._set_mu(mu)
            if self.restriction_method == 'submesh':
                r = df.assemble(self.form)
                if self.dirichlet_bc:
                    self.dirichlet_bc.apply(r)
                return self.range.make_array([r])
            elif self.restriction_method == 'IQP':
                assert self.quadrature_range is not None
                g = local_project(self.form, self.quadrature_range.V)
                g_vec = g.vector()
                if self.dirichlet_bc:
                    self.dirichlet_bc.apply(g_vec)
                return self.quadrature_range.make_array([g_vec])
            else:
                assert False

        def restricted(self, dofs):
            self.logger.info(f"Using restriction method {self.restriction_method} ...")
            if self.restriction_method == 'submesh':
                # first determine affected cells
                self.logger.info('Computing affected cells ...')
                mesh = self.range.V.mesh()
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

                self.logger.info("Building submesh ...")
                subdomain = df.MeshFunction("size_t", mesh, mesh.geometry().dim())
                for ci in affected_cell_indices:
                    subdomain.set_value(ci, 1)
                submesh = df.SubMesh(mesh, subdomain, 1)

                self.logger.info("Building ufl form on submesh ...")
                V_r_range = df.FunctionSpace(submesh, self.range.V.ufl_element())
                args = tuple(
                    (df.function.argument.Argument(V_r_range, arg.number(), arg.part())
                     if arg.ufl_function_space() == self.range.V else arg)
                    for arg in self.form.arguments()
                )
                form_r = ufl.replace_integral_domains(
                    self.form(*args, coefficients={}),
                    submesh.ufl_domain()
                )

                if self.dirichlet_bc:
                    bc = self.dirichlet_bc
                    if not bc.user_subdomain():
                        raise NotImplementedError
                    bc_r = df.DirichletBC(V_r_range, bc.value(), bc.user_subdomain(), bc.method())
                else:
                    bc_r = None

                self.logger.info("Computing range dof mapping ...")
                u = df.Function(self.range.V)
                u_vec = u.vector()
                restricted_range_dofs = []
                for range_dof in dofs:# dofs = interpolation dofs from original mesh
                    u_vec.zero()
                    u_vec[range_dof] = 1
                    u_r = df.interpolate(u, V_r_range)
                    u_r = u_r.vector().get_local()
                    if not np.all(np.logical_or(np.abs(u_r) < 1e-10, np.abs(u_r - 1.) < 1e-10)):
                        raise NotImplementedError
                    r_dof = np.where(np.abs(u_r - 1.) < 1e-10)[0]
                    if not len(r_dof) == 1:
                        raise NotImplementedError
                    restricted_range_dofs.append(r_dof[0])
                restricted_range_dofs = np.array(restricted_range_dofs, dtype=np.int32)

                op_r = FenicsNonaffineVectorOperator(form_r,
                                                     FenicsVectorSpace(V_r_range),
                                                     dirichlet_bc=bc_r,
                                                     parameter_setter=self.parameter_setter,
                                                     parameter_type=self.parameter_type)
                return (RestrictedFenicsNonaffineVectorOperatorSubmesh(op_r, restricted_range_dofs), [0])

            elif self.restriction_method == 'IQP':
                Q = self.quadrature_range.V
                assert isinstance(Q, df.FunctionSpace)
                #  assert isinstance(self.form, types.FunctionType)
                quad_points = Q.tabulate_dof_coordinates()
                points = quad_points[dofs]
                range_space = NumpyVectorSpace(points.size)
                # somehow get numpy version of the form(?)
                op_r = FenicsNonaffineVectorOperator(self.form,
                                                     range_space,
                                                     parameter_setter=self.parameter_setter,
                                                     parameter_type=self.parameter_type,
                                                     restriction_method=self.restriction_method,
                                                     numpy_form=self.numpy_form)
                return (RestrictedFenicsNonaffineVectorOperatorIQP(op_r, points), [0])

            else:
                assert False

    class RestrictedFenicsNonaffineVectorOperatorSubmesh(OperatorBase):
        """Restricted version of FenicsNonaffineVectorOperator"""

        linear = True
        source = NumpyVectorSpace(1)

        def __init__(self, operator, restricted_range_dofs):
            self.range = NumpyVectorSpace(len(restricted_range_dofs))
            self.op = operator
            self.restricted_range_dofs = restricted_range_dofs
            self.build_parameter_type(operator)

        def apply(self, U, mu=None):
            # TODO: application to len(U) > 1?, see FenicsOperator
            assert U in self.source
            self.op._set_mu(mu)
            F = df.assemble(self.op.form)
            if self.op.dirichlet_bc:
                self.op.dirichlet_bc.apply(F)
            return self.range.make_array([F[self.restricted_range_dofs]])

    class RestrictedFenicsNonaffineVectorOperatorIQP(OperatorBase):
        """Restricted version of FenicsNonaffineVectorOperator"""

        linear = True
        source = NumpyVectorSpace(1)

        def __init__(self, operator, points):
            assert isinstance(points, np.ndarray)
            assert points.ndim in (2, 3), NotImplementedError(
                "Geometric dimension should be 2 or 3"
            )
            self.range = NumpyVectorSpace(points[:, 0].size)
            self.op = operator
            self.points = points
            self.build_parameter_type(operator)

        def apply(self, U, mu=None):
            assert U in self.source
            self.op._set_mu(mu)
            F = self.op.numpy_form(self.points, mu)
            return self.range.make_array([F])

    @defaults('solver', 'preconditioner')
    def _solver_options(solver='bicgstab', preconditioner='amg'):
        return {'solver': solver, 'preconditioner': preconditioner}

    def _apply_inverse(matrix, r, v, options=None):
        options = options or _solver_options()
        solver = options.get('solver')
        preconditioner = options.get('preconditioner')
        # preconditioner argument may only be specified for iterative solvers:
        options = (solver, preconditioner) if preconditioner else (solver,)
        df.solve(matrix, r, v, *options)
