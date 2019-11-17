# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


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
    from pymor.operators.numpy import NumpyMatrixOperator
    from pymor.vectorarrays.interfaces import _create_random_values
    from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorSpace
    from pymor.vectorarrays.numpy import NumpyVectorSpace

    class FenicsOperator(OperatorBase):
        """Wraps a FEniCS form as an |Operator|."""

        linear = False

        @defaults('restriction_method')
        def __init__(self, form, source_space, range_space, source_function, dirichlet_bc=None,
                     parameter_setter=None, parameter_type=None, solver_options=None,
                     restriction_method='submesh', name=None, material=None, subdomain_data=None,
                     linearized_form=False):
            assert restriction_method in ('assemble_local', 'submesh')
            if linearized_form:
                assert len(form.arguments()) == 2
                # form = int eps·C·eps dx - int v·t ds + int eps·sigma dx == 0
                self.J = df.lhs(form)# will be used as jacobian
                form = - df.rhs(form)# will be used as operator, note the minus sign ... see
                #  newton.py line 93
                self.linearized_form = linearized_form
            else:
                assert len(form.arguments()) == 1
            if material:
                assert inspect.isclass(type(material))
            if subdomain_data:# TODO: this is not really needed
                assert restriction_method == 'assemble_local'# for now submesh does not work with
                #  meso framework
            assert isinstance(dirichlet_bc, list) or dirichlet_bc == None
            self.form = form
            self.source = source_space
            self.range = range_space
            self.source_function = source_function
            self.dirichlet_bc = dirichlet_bc
            self.parameter_setter = parameter_setter
            self.build_parameter_type(parameter_type)
            self.solver_options = solver_options
            self.restriction_method = restriction_method
            self.name = name
            self.material = material

        def _set_mu(self, mu=None):
            mu = self.parse_parameter(mu)
            if self.parameter_setter:
                self.parameter_setter(mu)

        def apply(self, U, mu=None):
            assert U in self.source
            self._set_mu(mu)
            R = []
            source_vec = self.source_function.vector()
            for u in U._list:
                source_vec[:] = u.impl
                if self.material:
                    self.material.update_history(u.impl, self.range)
                r = df.assemble(self.form)
                if self.dirichlet_bc:
                    for bc in self.dirichlet_bc:
                        bc.apply(r, source_vec)
                R.append(r)
            return self.range.make_array(R)

        def apply_unassembled(self, U, mu=None):
            assert U in self.source
            self._set_mu(mu)
            R = []
            source_vec = self.source_function.vector()
            for u in U._list:
                source_vec[:] = u.impl
                if self.material:
                    self.material.update_history(u.impl, self.range)

                le_vectors = []
                for cell in affected_cells:
                    le_vectors.append(df.assemble_local(self.form, cell))
                r = np.hstack(le_vectors)
                # TODO: apply bc to local element vectors?
                #  if self.dirichlet_bc:
                #      for bc in self.dirichlet_bc:
                #          bc.apply(r, source_vec)
                R.append(r)

            # ### range space for un-assembled vector
            space = self.range.V
            dim = space.mesh().num_cells() * space.dofmap().cell_dofs(0).shape[0]
            range_space = NumpyVectorSpace(dim)
            return range_space.make_array(R)

        def jacobian(self, U, mu=None):
            assert U in self.source and len(U) == 1
            self._set_mu(mu)
            source_vec = self.source_function.vector()
            source_vec[:] = U._list[0].impl
            #  if self.material:
                #  self.material.update_history(source_vec, self.range)
            if self.linearized_form:
                matrix = df.assemble(self.J)
            else:
                matrix = df.assemble(df.derivative(self.form, self.source_function))
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
                    RestrictedFenicsOperatorAssembleLocal(self, np.array(dofs), source_dofs.copy(), affected_cells,
                                                          source_local_restricted, range_local_restricted,
                                                          dir_dofs_r, dir_vals_r, dir_dofs_r_source),
                    source_dofs
                )

            elif self.restriction_method == 'submesh':
                # generate restricted spaces
                self.logger.info('Building submesh ...')
                subdomain = df.MeshFunction('size_t', mesh, mesh.geometry().dim())
                for ci in affected_cell_indices:
                    subdomain.set_value(ci, 1)
                submesh = df.SubMesh(mesh, subdomain, 1)

                # build restricted form
                self.logger.info('Building UFL form on submesh ...')
                V_r_source = df.FunctionSpace(submesh, self.source.V.ufl_element())
                V_r_range = df.FunctionSpace(submesh, self.range.V.ufl_element())
                assert V_r_source.dim() == len(source_dofs)

                if self.source.V != self.range.V:
                    assert all(arg.ufl_function_space() != self.source.V for arg in self.form.arguments())
                args = tuple((df.function.argument.Argument(V_r_range, arg.number(), arg.part())
                              if arg.ufl_function_space() == self.range.V else arg)
                             for arg in self.form.arguments())
                if any(isinstance(coeff, df.Function) and coeff != self.source_function for coeff in
                       self.form.coefficients()):
                    raise NotImplementedError
                source_function_r = df.Function(V_r_source)
                form_r = ufl.replace_integral_domains(
                    self.form(*args, coefficients={self.source_function: source_function_r}),
                    submesh.ufl_domain()
                )
                if self.dirichlet_bc:
                    bc = self.dirichlet_bc
                    if not bc.user_subdomain():
                        raise NotImplementedError
                    bc_r = df.DirichletBC(V_r_source, bc.value(), bc.user_subdomain(), bc.method())
                else:
                    bc_r = None

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

                # source dof mapping
                self.logger.info('Computing range DOF mapping ...')
                u = df.Function(self.range.V)
                u_vec = u.vector()
                restricted_range_dofs = []
                for range_dof in dofs:
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

                op_r = FenicsOperator(form_r, FenicsVectorSpace(V_r_source), FenicsVectorSpace(V_r_range),
                                      source_function_r, dirichlet_bc=bc_r, parameter_setter=self.parameter_setter,
                                      parameter_type=self.parameter_type)

                return (RestrictedFenicsOperatorSubMesh(op_r, restricted_range_dofs),
                        source_dofs[np.argsort(restricted_source_dofs)])
            else:
                assert False

    class RestrictedFenicsOperatorAssembleLocal(OperatorBase):

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
            source_vec = operator.source_function.vector()
            operator._set_mu(mu)
            R = np.zeros((len(U), self.range.dim + 1))
            for u, r in zip(U.data, R):
                source_vec[self.source_dofs] = u
                if self.operator.material:
                    self.operator.material.update_history(u, self.operator.range,
                                                          source_dofs=self.source_dofs)
                for cell, local_restricted in zip(self.cells, self.range_local_restricted):
                    local_evaluations = df.assemble_local(operator.form, cell)
                    r[local_restricted] += local_evaluations
                if self.dirichlet_values:
                    r[self.dirichlet_dofs] = u[self.dirichlet_source_dofs] - self.dirichlet_values
            return self.range.make_array(R[:, :-1])

        def jacobian(self, U, mu=None):
            assert U in self.source and len(U) == 1
            operator = self.operator
            source_vec = operator.source_function.vector()
            operator._set_mu(mu)
            J = np.zeros((self.range.dim + 1, self.source.dim + 1))
            source_vec[self.source_dofs] = U.data[0]
            for cell, range_local_restricted, source_local_restricted in zip(self.cells,
                                                                             self.range_local_restricted,
                                                                             self.source_local_restricted):
                if self.operator.material and self.operator.linearized_form:
                    local_matrix = df.assemble_local(operator.J, cell)
                else:
                    local_matrix = df.assemble_local(df.derivative(operator.form, operator.source_function), cell)
                #  J[np.meshgrid(range_local_restricted, source_local_restricted, indexing='ij')] += local_matrix
                J[tuple(np.meshgrid(range_local_restricted, source_local_restricted, indexing='ij'))] += local_matrix
            if self.dirichlet_dofs is not None:
                J[self.dirichlet_dofs, :] = 0.# if self.dirichlet_dofs is None, don't do this. this will
                #  set all elements of J to zero
                #  J[np.meshgrid(self.dirichlet_dofs, self.dirichlet_source_dofs, indexing='ij')] = 1.
                J[tuple(np.meshgrid(self.dirichlet_dofs, self.dirichlet_source_dofs, indexing='ij'))] = 1.
            return NumpyMatrixOperator(J[:-1, :-1])

        def restricted(self, dofs):
            raise NotImplementedError

    class RestrictedFenicsOperatorSubMesh(OperatorBase):

        linear = False

        def __init__(self, op, restricted_range_dofs):
            self.source = NumpyVectorSpace(op.source.dim)
            self.range = NumpyVectorSpace(len(restricted_range_dofs))
            self.op = op
            self.restricted_range_dofs = restricted_range_dofs
            self.build_parameter_type(op)

        def apply(self, U, mu=None):
            assert U in self.source
            UU = self.op.source.zeros(len(U))
            for uu, u in zip(UU._list, U.data):
                uu.impl[:] = u
            VV = self.op.apply(UU, mu=mu)
            V = self.range.zeros(len(VV))
            for v, vv in zip(V.data, VV._list):
                v[:] = vv.impl[self.restricted_range_dofs]
            return V

        def jacobian(self, U, mu=None):
            assert U in self.source and len(U) == 1
            UU = self.op.source.zeros()
            UU._list[0].impl[:] = U.data[0]
            JJ = self.op.jacobian(UU, mu=mu)
            return NumpyMatrixOperator(JJ.matrix.array()[self.restricted_range_dofs, :])

    @defaults('solver', 'preconditioner')
    def _solver_options(solver='bicgstab', preconditioner='amg'):
        return {'solver': solver, 'preconditioner': preconditioner}
    #  @defaults('solver', 'preconditioner')
    #  def _solver_options(solver='cg', preconditioner='ilu'):
    #      return {'solver': solver, 'preconditioner': preconditioner}

    def _apply_inverse(matrix, r, v, options=None):
        options = options or _solver_options()
        solver = options.get('solver')
        preconditioner = options.get('preconditioner')
        # preconditioner argument may only be specified for iterative solvers:
        options = (solver, preconditioner) if preconditioner else (solver,)
        df.solve(matrix, r, v, *options)

    class FenicsVisualizer(BasicInterface):
        """Visualize a FEniCS grid function.

        Parameters
        ----------
        space
            The `FenicsVectorSpace` for which we want to visualize DOF vectors.
        mesh_refinements
            Number of uniform mesh refinements to perform for vtk visualization
            (of functions from higher-order FE spaces).
        """

        def __init__(self, space, mesh_refinements=0):
            self.space = space
            self.mesh_refinements = mesh_refinements

        def visualize(self, U, m, title='', legend=None, filename=None, block=True,
                      separate_colorbars=True):
            """Visualize the provided data.

            Parameters
            ----------
            U
                |VectorArray| of the data to visualize (length must be 1). Alternatively,
                a tuple of |VectorArrays| which will be visualized in separate windows.
                If `filename` is specified, only one |VectorArray| may be provided which,
                however, is allowed to contain multipled vectors that will be interpreted
                as a time series.
            m
                Filled in by :meth:`pymor.models.ModelBase.visualize` (ignored).
            title
                Title of the plot.
            legend
                Description of the data that is plotted. If `U` is a tuple of |VectorArrays|,
                `legend` has to be a tuple of the same length.
            filename
                If specified, write the data to that file. `filename` needs to have an extension
                supported by FEniCS (e.g. `.pvd`).
            separate_colorbars
                If `True`, use separate colorbars for each subplot.
            block
                If `True`, block execution until the plot window is closed.
            """
            if filename:
                name, ext = filename.split('.')
                if ext == 'xdmf':# add all VectorArrays to one xdmf file
                    if self.mesh_refinements:
                        raise NotImplementedError
                    assert U in self.space and len(U) == 1 or isinstance(U, tuple)
                    assert legend is not None
                    if not isinstance(U, tuple):
                        U = (U,)
                    if isinstance(legend, str):
                        legend = (legend,)
                    f = df.XDMFFile(filename)
                    f.parameters["functions_share_mesh"] = True
                    f.parameters["rewrite_function_mesh"] = False
                    f.parameters["flush_output"] = True

                    for (array, l) in zip(U, legend):
                        function = df.Function(self.space.V, name=l)
                        for (time, u) in enumerate(array._list):
                            function.vector()[:] = u.impl
                            f.write(function, time)

                else:
                    assert not isinstance(U, tuple)
                    assert U in self.space
                    f = df.File(filename)
                    coarse_function = df.Function(self.space.V)
                    if self.mesh_refinements:
                        mesh = self.space.V.mesh()
                        for _ in range(self.mesh_refinements):
                            mesh = df.refine(mesh)
                        V_fine = df.FunctionSpace(mesh, self.space.V.ufl_element())
                        function = df.Function(V_fine)
                    else:
                        function = coarse_function
                    if legend:
                        function.rename(legend, legend)
                    for u in U._list:
                        coarse_function.vector()[:] = u.impl
                        if self.mesh_refinements:
                            function.vector()[:] = df.interpolate(coarse_function, V_fine).vector()
                        f << function
            else:
                from matplotlib import pyplot as plt

                assert U in self.space and len(U) == 1 \
                    or (isinstance(U, tuple) and all(u in self.space for u in U) and all(len(u) == 1 for u in U))
                if not isinstance(U, tuple):
                    U = (U,)
                if isinstance(legend, str):
                    legend = (legend,)
                assert legend is None or len(legend) == len(U)

                if not separate_colorbars:
                    vmin = np.inf
                    vmax = -np.inf
                    for u in U:
                        vec = u._list[0].impl
                        vmin = min(vmin, vec.min())
                        vmax = max(vmax, vec.max())

                for i, u in enumerate(U):
                    function = df.Function(self.space.V)
                    function.vector()[:] = u._list[0].impl
                    if legend:
                        tit = title + ' -- ' if title else ''
                        tit += legend[i]
                    else:
                        tit = title
                    if separate_colorbars:
                        plt.figure()
                        df.plot(function, title=tit)
                    else:
                        plt.figure()
                        df.plot(function, title=tit,
                                range_min=vmin, range_max=vmax)
                plt.show(block=block)

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
            assert U in self.source# U is scalar, but does not influence the result?!
            self._set_mu(mu)
            if self.restriction_method == 'submesh':
                r = df.assemble(self.form)
                if self.dirichlet_bc:
                    for bc in self.dirichlet_bc:
                        bc.apply(r)
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
