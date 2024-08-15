# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional, Union

import dolfinx as df
import numpy as np
import numpy.typing as npt
import ufl
from dolfinx.fem.petsc import (
    apply_lifting,
    assemble_matrix,
    assemble_vector,
    create_matrix,
    create_vector,
    set_bc,
)
from petsc4py import PETSc

from pymor.core.base import ImmutableObject
from pymor.core.config import config
from pymor.core.defaults import defaults
from pymor.core.pickle import unpicklable
from pymor.operators.constructions import VectorFunctional, VectorOperator
from pymor.operators.interface import Operator
from pymor.operators.list import LinearComplexifiedListVectorArrayOperatorBase
from pymor.vectorarrays.interface import _create_random_values
from pymor.vectorarrays.list import (
    ComplexifiedListVectorSpace,
    ComplexifiedVector,
    CopyOnWriteVector,
)
from pymor.vectorarrays.numpy import NumpyVectorSpace

config.require('FENICSX')


class BCGeom(NamedTuple):
    """Dirichlet BC defined geometrically."""

    value: Union[df.fem.Function, df.fem.Constant, npt.NDArray[Any]]
    locator: Callable
    V: df.fem.FunctionSpace


class BCTopo(NamedTuple):
    """Dirichlet BC defined topologically."""

    value: Union[df.fem.Function, df.fem.Constant, npt.NDArray[Any]]
    entities: npt.NDArray[np.int32]
    entity_dim: int
    V: df.fem.FunctionSpace
    sub: Optional[int] = None


class SubmeshWrapper(NamedTuple):
    mesh: df.mesh.Mesh
    parent_entities: list[int]
    vertex_map: list[int]
    geom_map: list[int]


def _assemble_petsc_mat(A: PETSc.Mat, acpp, bcs):
    A.zeroEntries()
    assemble_matrix(A, acpp, bcs=bcs)
    A.assemble()

def _assemble_petsc_vec(b: PETSc.Vec, Lcpp, bcs, acpp=None):
    with b.localForm() as b_loc:
        b_loc.set(0)
    assemble_vector(b, Lcpp)
    # apply lifting?
    if acpp is not None:
        apply_lifting(b, [acpp], bcs=[bcs])

    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    set_bc(b, bcs)

def _create_dirichlet_bcs(bcs: tuple[Union[BCGeom, BCTopo], ...]) -> list[df.fem.DirichletBC]:
    """Creates list of `df.fem.DirichletBC`.

    Args:
        bcs: The BC specifications.
    """
    r = list()
    for nt in bcs:
        space = None
        if isinstance(nt, BCGeom):
            space = nt.V
            dofs = df.fem.locate_dofs_geometrical(space, nt.locator)
        elif isinstance(nt, BCTopo):
            space = nt.V.sub(nt.sub) if nt.sub is not None else nt.V
            dofs = df.fem.locate_dofs_topological(space, nt.entity_dim, nt.entities)
        else:
            raise TypeError
        try:
            bc = df.fem.dirichletbc(nt.value, dofs, space)
        except TypeError:
            bc = df.fem.dirichletbc(nt.value, dofs)
        r.append(bc)
    return r


# from ufl/finiteelement/form.py
# this method not exist anymore for ufl version > 2023.1.1
def _replace_integral_domains(form, common_domain):
    """Replaces integral domains for form.

    Given a form and a domain, assign a common integration domain to
    all integrals.
    Does not modify the input form (``Form`` should always be
    immutable).  This is to support ill formed forms with no domain
    specified, sometimes occurring in pydolfin, e.g. assemble(1*dx,
    mesh=mesh).
    """
    domains = form.ufl_domains()
    if common_domain is not None:
        gdim = common_domain.geometric_dimension()
        tdim = common_domain.topological_dimension()
        if not all(
            (gdim == domain.geometric_dimension() and tdim == domain.topological_dimension()) for domain in domains
        ):
            raise ValueError('Common domain does not share dimensions with form domains.')

    reconstruct = False
    integrals = []
    for itg in form.integrals():
        domain = itg.ufl_domain()
        if domain != common_domain:
            itg = itg.reconstruct(domain=common_domain)
            reconstruct = True
        integrals.append(itg)
    if reconstruct:
        form = ufl.Form(integrals)
    return form


def _restrict_form(form, S, R, submesh: SubmeshWrapper, padding=1e-14):
    """Restrict `form` to submesh.

    Args:
        form: The UFL form to restrict.
        S: Source space.
        R: Range space.
        submesh: The submesh for restricted evaluation.
        padding: Padding parameter for creation of interpolation data.
    """
    # TODO: replace !=, == checks with V.element?

    # Note that for some
    # V = fem.functionspace(domain, S.ufl_element())
    # W = fem.functionspace(domain, R.ufl_element())
    # with S and R being the same object (R = S)
    # V == W returns False

    # If S == R returns True, this leads to the result that each argument
    # in `args` below, will be an element of the space R, which may
    # not be desired.
    # Therefore, V_r_range = V_r_source is set if S == R.

    V_r_source = df.fem.functionspace(submesh.mesh, S.ufl_element())
    if S != R:
        assert all(arg.ufl_function_space() != S for arg in form.arguments())
        V_r_range = df.fem.functionspace(submesh.mesh, R.ufl_element())
    else:
        V_r_range = V_r_source

    interp_data = df.fem.create_nonmatching_meshes_interpolation_data(
        V_r_source.mesh, V_r_source.element, S.mesh, padding=padding
    )

    args = tuple(
        (
            df.fem.function.ufl.argument.Argument(V_r_range, arg.number(), arg.part())
            if arg.ufl_function_space() == R
            else arg
        )
        for arg in form.arguments()
    )

    new_coeffs = {}
    for function in form.coefficients():
        # replace coefficients (fem.Function)
        name = function.name
        if function.function_space != S:
            raise NotImplementedError(
                    'Restriction of coefficients that are not elements of the source space '
                    'of the full operator is not supported.'
            )
        new_coeffs[function] = df.fem.Function(V_r_source, name=name)
        new_coeffs[function].interpolate(function, nmm_interpolation_data=interp_data)

    form_r = _replace_integral_domains(form(*args, coefficients=new_coeffs), submesh.mesh.ufl_domain())

    return form_r, V_r_source, V_r_range, interp_data


def _build_dof_map(V, V_r, dofs, interp_data) -> npt.NDArray[np.int32]:
    """Computes dofs in `V_r` correpsonding to dofs in V.

    Args:
        V: Full space.
        V_r: Restricted space.
        dofs: Magic dofs.
        interp_data: Interpolation data for non-matching meshes.
    """
    # TODO: do not create fem.Function inside function
    u = df.fem.Function(V)
    u_vec = u.x.petsc_vec

    u_r = df.fem.Function(V_r)
    u_r_vec = u_r.x.petsc_vec

    restricted_dofs = []
    for dof in dofs:
        u_vec.zeroEntries()
        u_vec.array[dof] = 1
        u_r.interpolate(u, nmm_interpolation_data=interp_data)
        u_r_array = u_r_vec.array
        if not np.all(np.logical_or(np.abs(u_r_array) < 1e-10, np.abs(u_r_array - 1.0) < 1e-10)):
            raise NotImplementedError
        r_dof = np.where(np.abs(u_r_array - 1.0) < 1e-10)[0]
        if not len(r_dof) == 1:
            raise NotImplementedError
        restricted_dofs.append(r_dof[0])
    restricted_dofs = np.array(restricted_dofs, dtype=np.int32)
    assert len(set(restricted_dofs)) == len(set(dofs))
    return restricted_dofs


def _affected_cells(V, dofs):
    """Returns affected cells.

    Args:
        V: The FE space.
        dofs: Interpolation dofs for restricted evaluation.
    """
    domain = V.mesh
    dofmap = V.dofmap

    affected_cells = set()
    num_cells = domain.topology.index_map(domain.topology.dim).size_local
    for cell in range(num_cells):
        cell_dofs = dofmap.cell_dofs(cell)
        for dof in cell_dofs:
            for b in range(dofmap.bs):
                if dof * dofmap.bs + b in dofs:
                    affected_cells.add(cell)
                    continue

    affected_cells = np.array(sorted(affected_cells), dtype=np.int32)
    return affected_cells


def _restrict_bc_topo(mesh, submesh_wrapper, bc, g, V_r):
    def locate_dirichlet_entities(
        mesh: df.mesh.Mesh, submesh_wrapper: SubmeshWrapper, entity_dim: int, dirichlet: npt.NDArray[np.int32]
    ):
        submesh, parent_cells, _, _ = submesh_wrapper
        tdim = submesh.topology.dim

        mesh.topology.create_connectivity(tdim, entity_dim)
        submesh.topology.create_connectivity(tdim, entity_dim)
        parent_c2e = mesh.topology.connectivity(tdim, entity_dim)
        cell2entity = submesh.topology.connectivity(tdim, entity_dim)

        entities = []

        for cell_index in range(submesh.topology.index_map(tdim).size_local):
            parent_ents = parent_c2e.links(parent_cells[cell_index])
            child_ents = cell2entity.links(cell_index)

            for pent, cent in zip(parent_ents, child_ents):
                if pent in dirichlet:
                    entities.append(cent)

        return np.array(entities, dtype=np.int32)

    dim = bc.entity_dim
    tags = locate_dirichlet_entities(mesh, submesh_wrapper, dim, bc.entities)
    return BCTopo(g, tags, dim, V_r, sub=bc.sub)


@unpicklable
class FenicsxVector(CopyOnWriteVector):
    """Wraps a FEniCSx vector to make it usable with ListVectorArray."""

    def __init__(self, impl):
        self.impl = impl

    @classmethod
    def from_instance(cls, instance):
        return cls(instance.impl)

    def _copy_data(self):
        self.impl = self.impl.copy()

    def to_numpy(self, ensure_copy=False):
        # u.vector.array returns array of length n, if n is the number of dofs on the process
        # u.vector.array is not a copy
        # what is the desired behaviour in parallel? in the legacy fenics .get_local()
        # would return only dofs on local process, so this is fine I think
        return self.impl.array.copy() if ensure_copy else self.impl.array  # TODO: what happens here in parallel?

    def _scal(self, alpha):
        self.impl *= alpha

    def _axpy(self, alpha, x):
        if x is self:
            self.scal(1.0 + alpha)
        else:
            self.impl.axpy(alpha, x.impl)

    def inner(self, other):
        return self.impl.dot(other.impl)

    def norm(self):
        # u.vector.norm(PETSc.NormType.NORM_2) returns global norm
        # same behaviour as in legacy fenics
        return self.impl.norm(PETSc.NormType.NORM_2)

    def norm2(self):
        return self.impl.norm(PETSc.NormType.NORM_2) ** 2

    def sup_norm(self):
        return self.impl.norm(PETSc.NormType.NORM_INFINITY)

    def dofs(self, dof_indices):
        dof_indices = np.array(dof_indices, dtype=np.intc)
        if len(dof_indices) == 0:
            return np.array([], dtype=np.intc)
        owned = self.impl.getOwnershipRange()
        in_range = np.isin(dof_indices, np.arange(*owned, dtype=np.intc))
        dofs_on_proc = dof_indices[in_range]
        # returns dof values for dof indices on process
        return self.impl.getValues(dofs_on_proc)

    def amax(self):
        raise NotImplementedError  # is implemented for complexified vector

    def __add__(self, other):
        return FenicsxVector(self.impl + other.impl)

    def __iadd__(self, other):
        self._copy_data_if_needed()
        self.impl += other.impl
        return self

    __radd__ = __add__

    def __sub__(self, other):
        return FenicsxVector(self.impl - other.impl)

    def __isub__(self, other):
        self._copy_data_if_needed()
        self.impl -= other.impl
        return self

    def __mul__(self, other):
        return FenicsxVector(self.impl * other)

    def __neg__(self):
        return FenicsxVector(-self.impl)


class ComplexifiedFenicsxVector(ComplexifiedVector):
    def amax(self):
        if self.imag_part is None:
            A = np.abs(self.real_part.impl.array)
        else:
            A = np.abs(self.real_part.impl.array + self.imag_part.impl.array * 1j)

        # Notes on PETSc.Vec
        # .abs --> in-place absolute value for each entry
        # .local_size --> returns int
        # .max --> returns tuple[int, float]: (dof, dof_value)
        # there seems to be no way in the interface to compute amax without making a copy.

        # maybe use vector.abs() and negate afterwards?

        max_ind_on_rank = np.argmax(A)
        max_val_on_rank = A[max_ind_on_rank]
        from pymor.tools import mpi

        if not mpi.parallel:
            return max_ind_on_rank, max_val_on_rank
        else:
            max_global_ind_on_rank = max_ind_on_rank + self.real_part.impl.local_size
            comm = self.real_part.impl.comm
            comm_size = comm.Get_size()

            max_inds = np.empty(comm_size, dtype='i')
            comm.Allgather(np.array(max_global_ind_on_rank, dtype='i'), max_inds)

            max_vals = np.empty(comm_size, dtype=np.float64)
            comm.Allgather(np.array(max_val_on_rank), max_vals)

            i = np.argmax(max_vals)
            return max_inds[i], max_vals[i]


class FenicsxVectorSpace(ComplexifiedListVectorSpace):
    real_vector_type = FenicsxVector
    vector_type = ComplexifiedFenicsxVector

    def __init__(self, V, id='STATE'):
        self.__auto_init(locals())

    @property
    def dim(self):
        return self.V.dofmap.index_map.size_global * self.V.dofmap.index_map_bs

    def __eq__(self, other):
        return type(other) is FenicsxVectorSpace and self.V == other.V and self.id == other.id

    # since we implement __eq__, we also need to implement __hash__
    def __hash__(self):
        return id(self.V) + hash(self.id)

    def real_zero_vector(self):
        impl = df.la.create_petsc_vector(self.V.dofmap.index_map, self.V.dofmap.index_map_bs)
        return FenicsxVector(impl)

    def real_full_vector(self, value):
        v = self.real_zero_vector()
        v.impl.set(value)
        return v

    def real_random_vector(self, distribution, **kwargs):
        v = self.real_zero_vector()
        values = _create_random_values(self.dim, distribution, **kwargs)  # TODO: parallel?
        v.to_numpy()[:] = values
        return v

    def real_vector_from_numpy(self, data, ensure_copy=False):
        v = self.real_zero_vector()
        v.to_numpy()[:] = data
        return v

    def real_make_vector(self, obj):
        return FenicsxVector(obj)


class FenicsxMatrixOperator(LinearComplexifiedListVectorArrayOperatorBase):
    """Wraps a FEniCSx matrix as an |Operator|."""

    def __init__(self, matrix, source_space, range_space, solver_options=None, name=None):
        self.__auto_init(locals())
        self.source = FenicsxVectorSpace(source_space)
        self.range = FenicsxVectorSpace(range_space)

    def _solver_options(self, adjoint=False):
        if adjoint:
            options = self.solver_options.get('inverse_adjoint') if self.solver_options else None
            if options is None:
                options = self.solver_options.get('inverse') if self.solver_options else None
        else:
            options = self.solver_options.get('inverse') if self.solver_options else None
        return options or _solver_options()

    def _create_solver(self, adjoint=False):
        options = self._solver_options(adjoint)
        if adjoint:
            try:
                matrix = self._matrix_transpose
            except AttributeError as e:
                raise RuntimeError('_create_solver called before _matrix_transpose has been initialized.') from e
        else:
            matrix = self.matrix
        method = options.get('solver')
        preconditioner = options.get('preconditioner')
        solver = PETSc.KSP().create(self.source.V.mesh.comm)
        solver.setOperators(matrix)
        solver.setType(method)
        solver.getPC().setType(preconditioner)
        return solver

    def _apply_inverse(self, r, v, adjoint=False):
        try:
            solver = self._adjoint_solver if adjoint else self._solver
        except AttributeError:
            solver = self._create_solver(adjoint)
        solver.solve(v, r)
        if _solver_options()['keep_solver']:
            if adjoint:
                self._adjoint_solver = solver
            else:
                self._solver = solver

    def _real_apply_one_vector(self, u, mu=None, prepare_data=None):
        r = self.range.real_zero_vector()
        self.matrix.mult(u.impl, r.impl)
        return r

    def _real_apply_adjoint_one_vector(self, v, mu=None, prepare_data=None):
        r = self.source.real_zero_vector()
        self.matrix.multTranspose(v.impl, r.impl)
        return r

    def _real_apply_inverse_one_vector(self, v, mu=None, initial_guess=None, least_squares=False, prepare_data=None):
        if least_squares:
            raise NotImplementedError
        r = self.source.real_zero_vector() if initial_guess is None else initial_guess.copy(deep=True)
        self._apply_inverse(r.impl, v.impl)
        return r

    def _real_apply_inverse_adjoint_one_vector(
        self, u, mu=None, initial_guess=None, least_squares=False, prepare_data=None
    ):
        if least_squares:
            raise NotImplementedError
        r = self.range.real_zero_vector() if initial_guess is None else initial_guess.copy(deep=True)

        # since dolfin does not have "apply_inverse_adjoint", we assume
        # PETSc is used as backend and transpose the matrix
        if not hasattr(self, '_matrix_transpose'):
            self._matrix_transpose = PETSc.Mat()
            self.matrix.transpose(self._matrix_transpose)
        self._apply_inverse(r.impl, u.impl, adjoint=True)
        return r

    def _assemble_lincomb(self, operators, coefficients, identity_shift=0.0, solver_options=None, name=None):
        if not all(isinstance(op, FenicsxMatrixOperator) for op in operators):
            return None
        if identity_shift != 0:
            return None
        if np.iscomplexobj(coefficients):
            return None

        if coefficients[0] == 1:
            matrix = operators[0].matrix.copy()
        else:
            matrix = operators[0].matrix * coefficients[0]
        for op, c in zip(operators[1:], coefficients[1:]):
            matrix.axpy(c, op.matrix)
            # in general, we cannot assume the same nonzero pattern for
            # all matrices. how to improve this?

        return FenicsxMatrixOperator(matrix, self.source.V, self.range.V, solver_options=solver_options, name=name)


@defaults('solver', 'preconditioner', 'keep_solver')
def _solver_options(solver=PETSc.KSP.Type.PREONLY, preconditioner=PETSc.PC.Type.LU, keep_solver=True):
    return {'solver': solver, 'preconditioner': preconditioner, 'keep_solver': keep_solver}


class FenicsxVisualizer(ImmutableObject):
    """Visualize a FEniCSx grid function.

    Parameters
    ----------
    space
        The `FenicsVectorSpace` for which we want to visualize DOF vectors.
    """

    def __init__(self, space):
        self.space = space

    def visualize(self, U, title='', legend=None, filename=None, block=True, separate_colorbars=True):
        """Visualize the provided data.

        Parameters
        ----------
        U
            |VectorArray| of the data to visualize (length must be 1). Alternatively,
            a tuple of |VectorArrays| which will be visualized in separate windows.
            If `filename` is specified, only one |VectorArray| may be provided which,
            however, is allowed to contain multiple vectors that will be interpreted
            as a time series.
        title
            Title of the plot.
        legend
            Description of the data that is plotted. If `U` is a tuple of |VectorArrays|,
            `legend` has to be a tuple of the same length.
        filename
            If specified, write the data to that file. `filename` needs to have an extension
            supported by FEniCSx (e.g. `.xdmf`).
        separate_colorbars
            If `True`, use separate colorbars for each subplot.
        block
            If `True`, block execution until the plot window is closed.
        """
        if filename:
            assert not isinstance(U, tuple)
            assert U in self.space
            if block:
                self.logger.warning('visualize with filename!=None, block=True will not block')

            # TODO: add support for other formats? (VTKFile, FidesWriter)
            supported = ('.xdmf', '.bp')
            suffix = Path(filename).suffix
            if suffix not in supported:
                msg = (
                    'FenicsxVisualizer needs a filename with a suffix indicating a supported backend\n'
                    f'defaulting to .xdmf (possible choices: {supported})'
                )
                self.logger.warning(msg)
                suffix = '.xdmf'

            # ### Initialize output function
            domain = self.space.V.mesh
            output = df.fem.Function(self.space.V)
            if legend:
                output.rename(legend, legend)

            # ### Initialize outstream
            if suffix == '.xdmf':
                from dolfinx.io.utils import XDMFFile

                outstream = XDMFFile(domain.comm, Path(filename).with_suffix(suffix), 'w')
                outstream.write_mesh(domain)

                def write_output(t):
                    outstream.write_function(output, float(t))

            elif suffix == '.bp':
                from dolfinx.io.utils import VTXWriter

                # Paraview 5.11.2 crashes for engine='BP5'
                outstream = VTXWriter(domain.comm, Path(filename).with_suffix(suffix), [output], engine='BP4')

                def write_output(t):
                    outstream.write(float(t))

            else:
                raise NotImplementedError

            for time, u in enumerate(U.vectors):
                if u.imag_part is not None:
                    raise NotImplementedError
                output.x.petsc_vec[:] = u.real_part.impl[:]
                write_output(time)
            outstream.close()

        else:
            assert (
                U in self.space
                and len(U) == 1
                or (isinstance(U, tuple) and all(u in self.space for u in U) and all(len(u) == 1 for u in U))
            )
            if not isinstance(U, tuple):
                U = (U,)
            if isinstance(legend, str):
                legend = (legend,)
            if legend is None:
                legend = tuple(f'U{i}' for i in range(len(U)))
            assert legend is None or len(legend) == len(U)

            from pyvista import UnstructuredGrid
            from pyvista.plotting.plotter import Plotter

            # TODO: this does not work for VectorFunctionSpace

            rows = 1 if len(U) <= 2 else 2
            cols = int(np.ceil(len(U) / rows))
            plotter = Plotter(shape=(rows, cols))
            mesh_data = df.plot.vtk_mesh(self.space.V)
            for i, (u, l) in enumerate(zip(U, legend)):
                row = i // cols
                col = i - row * cols
                plotter.subplot(row, col)
                u_grid = UnstructuredGrid(*mesh_data)
                u_grid.point_data[l] = u.vectors[0].real_part.impl.array.real
                u_grid.set_active_scalars(l)
                plotter.add_mesh(u_grid, show_edges=False)
                plotter.add_scalar_bar(l)
                plotter.view_xy()
            plotter.show()


class FenicsxMatrixBasedOperator(Operator):
    """Wraps a parameterized FEniCSx linear or bilinear form as an |Operator|.

    Parameters
    ----------
    form
        The `ufl.Form` object which is assembled to a matrix or vector.
    params
        Dict mapping parameters to `dolfinx.fem.Constant` or dimension.
    param_setter
        Custom method to update all form coefficients to new parameter value.
        This is required if the form contains parametric `dolfinx.fem.Function`s.
    bcs
        Tuple of Dirichlet BCs.
    functional
        If `True` return a |VectorFunctional| instead of a |VectorOperator| in case
        `form` is a linear form.
    form_compiler_options
        FFCX Form compiler options. See `dolfinx.jit.ffcx_jit`.
    jit_options
        JIT compilation options. See `dolfinx.jit.ffcx_jit`.
    solver_options
        The |solver_options| for the assembled :class:`FenicsxMatrixOperator`.
    name
        Name of the operator.
    """

    linear = True

    def __init__(
        self,
        form: ufl.Form,
        params: dict,
        param_setter: Optional[Callable] = None,
        bcs: Optional[tuple[Union[BCGeom, BCTopo], ...]] = None,
        functional: Optional[bool] = False,
        form_compiler_options: Optional[dict] = None,
        jit_options: Optional[dict] = None,
        solver_options: Optional[dict] = None,
        name: Optional[str] = None,
    ):
        assert 1 <= len(form.arguments()) <= 2
        assert not functional or len(form.arguments()) == 1
        self.__auto_init(locals())
        if len(form.arguments()) == 2 or not functional:
            range_space = form.arguments()[0].ufl_function_space()
            self.range = FenicsxVectorSpace(range_space)
        else:
            self.range = NumpyVectorSpace(1)
        if len(form.arguments()) == 2 or functional:
            source_space = form.arguments()[0 if functional else 1].ufl_function_space()
            self.source = FenicsxVectorSpace(source_space)
        else:
            self.source = NumpyVectorSpace(1)
        parameters_own = {}
        for k, v in params.items():
            try:
                parameters_own[k] = v.value.size
            except AttributeError:
                parameters_own[k] = v
        self.parameters_own = parameters_own
        self._bcs = bcs or tuple()
        self.bcs = _create_dirichlet_bcs(bcs) if bcs is not None else list()
        self.compiled_form = df.fem.form(form, form_compiler_options=form_compiler_options, jit_options=jit_options)
        if len(form.arguments()) == 2:
            self.disc = create_matrix(self.compiled_form)
        else:
            self.disc = create_vector(self.compiled_form)

    def _set_mu(self, mu) -> None:
        assert self.parameters.assert_compatible(mu)
        if self.param_setter is None:
            # Assume params maps to `df.fem.Constant` only
            for name, constant in self.params.items():
                constant.value = mu[name]
        else:
            # if `self.form` contains parameter-dependent `df.fem.Function`s
            # the user needs to provide custom function how to update those
            self.param_setter(mu)

    def _assemble_matrix(self):
        _assemble_petsc_mat(self.disc, self.compiled_form, self.bcs)
        # self.disc.zeroEntries()
        # assemble_matrix(self.disc, self.compiled_form, bcs=self.bcs)
        # self.disc.assemble()

    def _assemble_vector(self):
        _assemble_petsc_vec(self.disc, self.compiled_form, self.bcs)
        # with self.disc.localForm() as b_loc:
        #     b_loc.set(0)
        # assemble_vector(self.disc, self.compiled_form)

        # TODO: Support inhomogeneous Dirichlet BCs (rhs)?
        # if self.form is linear and representing the rhs
        # then we would need the compiled form of the lhs in case of inhomogeneous Dirchlet BCs
        # apply_lifting(self.disc, [self.compiled_form_lhs], bcs=[bcs])

        # On second thought: If constant inhomogeneous Dirichlet BCs user
        # can always perform the dirichlet lift explicitly (model.deaffinize)

        # self.disc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        # set_bc(self.disc, self.bcs)

    def assemble(self, mu=None):
        self._set_mu(mu)

        if len(self.form.arguments()) == 2:
            self._assemble_matrix()
            return FenicsxMatrixOperator(
                self.disc,
                self.source.V,
                self.range.V,
                self.solver_options,
                self.name + '_assembled',
            )
        elif self.functional:
            self._assemble_vector()
            V = self.source.make_array([self.disc])
            return VectorFunctional(V)
        else:
            self._assemble_vector()
            V = self.range.make_array([self.disc])
            return VectorOperator(V)

    def apply(self, U, mu=None):
        return self.assemble(mu).apply(U)

    def apply_adjoint(self, V, mu=None):
        return self.assemble(mu).apply_adjoint(V)

    def as_range_array(self, mu=None):
        return self.assemble(mu).as_range_array()

    def as_source_array(self, mu=None):
        return self.assemble(mu).as_source_array()

    def apply_inverse(self, V, mu=None, initial_guess=None, least_squares=False):
        return self.assemble(mu).apply_inverse(V, initial_guess=initial_guess, least_squares=least_squares)

    def restricted(self, dofs, padding=1e-14):
        if len(self.form.arguments()) == 1:
            raise NotImplementedError

        # ### compute affected cells
        S = self.source.V
        R = self.range.V
        domain = S.mesh
        cells = _affected_cells(S, dofs)

        # ### compute source dofs based on affected cells
        source_dofmap = S.dofmap
        source_dofs = set()
        for cell_index in cells:
            local_dofs = source_dofmap.cell_dofs(cell_index)
            for ld in local_dofs:
                for b in range(source_dofmap.bs):
                    source_dofs.add(ld * source_dofmap.bs + b)
        source_dofs = np.array(sorted(source_dofs), dtype=dofs.dtype)

        # ### build submesh
        tdim = domain.topology.dim
        submesh = SubmeshWrapper(*df.mesh.create_submesh(domain, tdim, cells))

        # ### restrict form to submesh
        restricted_form, V_r_source, V_r_range, interp_data = _restrict_form(self.form, S, R, submesh, padding=padding)

        # ### restrict Dirichlet BCs
        r_bcs = list()
        for nt in self._bcs:
            if isinstance(nt.value, df.fem.Function):
                # interpolate function to submesh
                g = df.fem.Function(V_r_source, name=nt.value.name)
                g.interpolate(nt.value, nmm_interpolation_data=interp_data)
            else:
                # g is of type df.fem.Constant or np.ndarray
                g = nt.value

            if isinstance(nt, BCGeom):
                rbc = BCGeom(g, nt.locator, V_r_source)
            elif isinstance(nt, BCTopo):
                if nt.entity_dim == 1 and tdim == 3:
                    raise NotImplementedError
                rbc = _restrict_bc_topo(S.mesh, submesh, nt, g, V_r_source)
            else:
                raise TypeError
            r_bcs.append(rbc)
        r_bcs = tuple(r_bcs)

        # ### compute dof mapping source
        restricted_source_dofs = _build_dof_map(S, V_r_source, source_dofs, interp_data)

        # ### compute dof mapping range
        restricted_range_dofs = _build_dof_map(R, V_r_range, dofs, interp_data)

        # sanity checks
        assert source_dofs.size == V_r_source.dofmap.bs * V_r_source.dofmap.index_map.size_local
        assert restricted_source_dofs.size == source_dofs.size
        assert restricted_range_dofs.size == dofs.size

        # edge case: form has parametric coefficient
        if self.form.coefficients():
            assert self.param_setter is not None
            set_params = self.param_setter

            # TODO: Issue warning
            # The below code assumes that every coefficient in restricted form
            # is an element of source (S) and is interpolated to V_r_source
            # This results in PETSc ERROR Segmentation violation for functions of quadrature spaces!

            def param_setter(mu):
                set_params(mu)
                # in addition to original param setter
                # interpolate coefficients
                for r_coeff in restricted_form.coefficients():
                    for coeff in self.form.coefficients():
                        if r_coeff.name == coeff.name:
                            r_coeff.x.array[:] = 0.0
                            r_coeff.interpolate(coeff, nmm_interpolation_data=interp_data)
        else:
            param_setter = None

        op_r = FenicsxMatrixBasedOperator(
            restricted_form,
            self.params,
            param_setter=param_setter,
            bcs=r_bcs,
            functional=self.functional,
            form_compiler_options=self.form_compiler_options,
            jit_options=self.jit_options,
            solver_options=self.solver_options,
        )

        return (
            RestrictedFenicsxMatrixBasedOperator(op_r, restricted_range_dofs),
            source_dofs[np.argsort(restricted_source_dofs)],
        )


class RestrictedFenicsxMatrixBasedOperator(Operator):
    """Restricted :class:`FenicsxMatrixBasedOperator`."""

    linear = True

    def __init__(self, op, restricted_range_dofs):
        self.source = NumpyVectorSpace(op.source.dim)
        self.range = NumpyVectorSpace(len(restricted_range_dofs))
        self.op = op
        self.restricted_range_dofs = restricted_range_dofs

    def assemble(self, mu=None):
        operator = self.op.assemble(mu)
        return operator

    def apply(self, U, mu=None):
        assert U in self.source

        #  ### Old version
        # UU = self.op.source.zeros(len(U))
        # for uu, u in zip(UU.vectors, U.to_numpy()):
        #     uu.real_part.impl[:] = np.ascontiguousarray(u)
        # VV = self.op.apply(UU, mu=mu)
        # V = self.range.zeros(len(VV))
        # for v, vv in zip(V.to_numpy(), VV.vectors):
        #     v[:] = vv.real_part.impl[self.restricted_range_dofs]

        # NOTE
        # old version resulted in PETSc error: unassembled Vector

        UU = self.op.source.from_numpy(U.to_numpy())
        VV = self.op.apply(UU, mu=mu)
        V = self.range.make_array(VV.dofs(self.restricted_range_dofs))
        return V


class FenicsxOperator(Operator):
    """Wraps a FEniCSx form as an |Operator|.

    Args:
        F: The UFL Form.
        source_space: The source space.
        range_space: The range space.
        source_function: Solution function.
        dirichlet_bcs: Dirichlet BC definitions.
        J: UFL representation of the Jacobian.
        parameter_setter: Function to update parameters.
        parameters: Dict mapping parameters to `dolfinx.fem.Constant`.
        form_compiler_options: FFCX Form compiler options. See `dolfinx.jit.ffcx_jit`.
        jit_options: JIT compilation options. See `dolfinx.jit.ffcx_jit`.
        solver_options: Options for the newton solver.
        name: Name of the operator.
    """

    linear = False

    def __init__(self, F: ufl.Form, source_space: FenicsxVectorSpace,
                 range_space: FenicsxVectorSpace, source_function: df.fem.Function,
                 dirichlet_bcs: tuple[Union[BCGeom, BCTopo]]=(), J: Optional[ufl.Form] = None,
                 parameter_setter: Optional[Callable]=None,
                 parameters: Optional[dict]={}, form_compiler_options: Optional[dict] = None,
                 jit_options: Optional[dict] = None, solver_options: Optional[dict]=None,
                 name: Optional[str]=None):
        assert len(F.arguments()) == 1
        self.__auto_init(locals())
        self.source = source_space
        self.range = range_space
        self.parameters_own = parameters
        self._bcs = dirichlet_bcs
        self.bcs = _create_dirichlet_bcs(dirichlet_bcs)
        self._L = df.fem.form(F, form_compiler_options=form_compiler_options,
                                         jit_options=jit_options)
        if J is None:  # Create the Jacobian matrix, dF/du
            V = source_function.function_space
            du = ufl.TrialFunction(V)
            J = ufl.derivative(F, source_function, du)

        self._a = df.fem.form(J, form_compiler_options=form_compiler_options,
                                         jit_options=jit_options)
        # Note that `dolfinx.nls.petsc.NewtonSolver` usually manages
        # the matrix and vector objects
        self._A = create_matrix(self._a)
        self._b = create_vector(self._L)

    def _set_mu(self, mu=None):
        assert self.parameters.assert_compatible(mu)
        if self.parameter_setter:
            self.parameter_setter(mu)

    def _assemble_vector(self, x: PETSc.Vec):
        # follow conventions of `df.fem.petsc.NonlinearProblem.F`

        with self._b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(self._b, self._L)

        # Apply boundary condition
        apply_lifting(self._b, [self._a], bcs=[self.bcs], x0=[x], scale=-1.0)
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self._b, self.bcs, x, -1.0)

    def apply(self, U, mu=None):
        assert U in self.source
        self._set_mu(mu)
        R = []
        source_vec = self.source_function.x.petsc_vec
        for u in U.vectors:
            if u.imag_part is not None:
                raise NotImplementedError
            source_vec[:] = u.real_part.impl
            # TODO: what happens if the bcs are not inhomogeneous?
            # _assemble_petsc_vec(self._b, self._L, self.bcs, acpp=self._a)
            self._assemble_vector(source_vec)
            R.append(self._b.copy())
        return self.range.make_array(R)

    def jacobian(self, U, mu=None):
        assert U in self.source
        assert len(U) == 1
        if U.vectors[0].imag_part is not None:
            raise NotImplementedError
        self._set_mu(mu)
        source_vec = self.source_function.x.petsc_vec
        source_vec[:] = U.vectors[0].real_part.impl
        _assemble_petsc_mat(self._A, self._a, self.bcs)
        return FenicsxMatrixOperator(self._A, self.source.V, self.range.V)

    def restricted(self, dofs):
        raise NotImplementedError
        with self.logger.block(f'Restricting operator to {len(dofs)} dofs ...'):
            if len(dofs) == 0:
                return ZeroOperator(NumpyVectorSpace(0), NumpyVectorSpace(0)), np.array([], dtype=int)

            if self.source.V.mesh().id() != self.range.V.mesh().id():
                raise NotImplementedError

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
            affected_cell_indices = sorted(affected_cell_indices)

            if any(i.integral_type() not in ('cell', 'exterior_facet')
                   for i in self.form.integrals()):
                # enlarge affected_cell_indices if needed
                raise NotImplementedError

            self.logger.info('Computing source DOFs ...')
            source_dofmap = self.source.V.dofmap()
            source_dofs = set()
            for cell_index in affected_cell_indices:
                local_dofs = source_dofmap.cell_dofs(cell_index)
                source_dofs.update(local_dofs)
            source_dofs = np.array(sorted(source_dofs), dtype=np.intc)

            self.logger.info('Building submesh ...')
            subdomain = df.MeshFunction('size_t', mesh, mesh.geometry().dim())
            for ci in affected_cell_indices:
                subdomain.set_value(ci, 1)
            submesh = df.SubMesh(mesh, subdomain, 1)

            self.logger.info('Building UFL form on submesh ...')
            form_r, V_r_source, V_r_range, source_function_r = self._restrict_form(submesh, source_dofs)

            self.logger.info('Building DirichletBCs on submesh ...')
            bc_r = self._restrict_dirichlet_bcs(submesh, source_dofs, V_r_source)

            self.logger.info('Computing source DOF mapping ...')
            restricted_source_dofs = self._build_dof_map(self.source.V, V_r_source, source_dofs)

            self.logger.info('Computing range DOF mapping ...')
            restricted_range_dofs = self._build_dof_map(self.range.V, V_r_range, dofs)

            op_r = FenicsOperator(form_r, FenicsVectorSpace(V_r_source), FenicsVectorSpace(V_r_range),
                                  source_function_r, dirichlet_bcs=bc_r, parameter_setter=self.parameter_setter,
                                  parameters=self.parameters)

            return (RestrictedFenicsOperator(op_r, restricted_range_dofs),
                    source_dofs[np.argsort(restricted_source_dofs)])

    def _restrict_form(self, submesh, source_dofs):
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

        return form_r, V_r_source, V_r_range, source_function_r


class RestrictedFenicsxOperator(Operator):
    """Restricted :class:`FenicsxOperator`."""

    linear = False

    def __init__(self, op, restricted_range_dofs):
        self.source = NumpyVectorSpace(op.source.dim)
        self.range = NumpyVectorSpace(len(restricted_range_dofs))
        self.op = op
        self.restricted_range_dofs = restricted_range_dofs

    def apply(self, U, mu=None):
        assert U in self.source
        UU = self.op.source.zeros(len(U))
        for uu, u in zip(UU.vectors, U.to_numpy()):
            uu.real_part.impl[:] = np.ascontiguousarray(u)
        VV = self.op.apply(UU, mu=mu)
        V = self.range.zeros(len(VV))
        for v, vv in zip(V.to_numpy(), VV.vectors):
            v[:] = vv.real_part.impl[self.restricted_range_dofs]
        return V

    def jacobian(self, U, mu=None):
        assert U in self.source
        assert len(U) == 1
        UU = self.op.source.zeros()
        UU.vectors[0].real_part.impl[:] = np.ascontiguousarray(U.to_numpy()[0])
        JJ = self.op.jacobian(UU, mu=mu)
        return NumpyMatrixOperator(JJ.matrix.array()[self.restricted_range_dofs, :])
