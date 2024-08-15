#!/usr/bin/env python3
# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from typer import Argument, run

from pymor.core.config import config


def main(
    dim: int = Argument(..., help='Spatial dimension of the problem.'),
    n: int = Argument(..., help='Number of mesh intervals per spatial dimension.'),
    order: int = Argument(..., help='Finite element order.'),
):
    """Reduces a FEniCSx-based nonlinear diffusion problem using POD/DEIM."""
    from pymor.tools import mpi
    config.require('FENICSX')

    # TODO: test parallel execution. SubMesh now runs in parallel.
    if mpi.parallel:
        from pymor.models.mpi import mpi_wrap_model
        local_models = mpi.call(mpi.function_call_manage, discretize, dim, n, order)
        fom = mpi_wrap_model(local_models, use_with=True, pickle_local_spaces=False)
    else:
        fom = discretize(dim, n, order)

    parameter_space = fom.parameters.space((0, 1000.))

    # ### ROM generation (POD/DEIM)
    from pymor.algorithms.ei import ei_greedy
    from pymor.algorithms.newton import newton
    from pymor.algorithms.pod import pod
    from pymor.operators.ei import EmpiricalInterpolatedOperator
    from pymor.reductors.basic import StationaryRBReductor

    U = fom.solution_space.empty()
    residuals = fom.solution_space.empty()
    for mu in parameter_space.sample_uniformly(10):
        UU, data = newton(fom.operator, fom.rhs.as_vector(), mu=mu, rtol=1e-6, return_residuals=True)
        U.append(UU)
        residuals.append(data['residuals'])

    dofs, cb, _ = ei_greedy(residuals, rtol=1e-7)
    # TODO: implement FenicsxOperator.restricted
    ei_op = EmpiricalInterpolatedOperator(fom.operator, collateral_basis=cb, interpolation_dofs=dofs, triangular=True)

    rb, svals = pod(U, rtol=1e-7)
    fom_ei = fom.with_(operator=ei_op)
    reductor = StationaryRBReductor(fom_ei, rb)
    rom = reductor.reduce()
    # the reductor currently removes all solver_options so we need to add them again
    rom = rom.with_(operator=rom.operator.with_(solver_options=fom.operator.solver_options))

    # ### ROM validation
    import time

    import numpy as np

    errs = []
    speedups = []
    for mu in parameter_space.sample_randomly(10):
        tic = time.perf_counter()
        U = fom.solve(mu)
        t_fom = time.perf_counter() - tic

        tic = time.perf_counter()
        u_red = rom.solve(mu)
        t_rom = time.perf_counter() - tic

        U_red = reductor.reconstruct(u_red)
        errs.append(((U - U_red).norm() / U.norm())[0])
        speedups.append(t_fom / t_rom)
    print(f'Maximum relative ROM error: {max(errs)}')
    print(f'Median of ROM speedup: {np.median(speedups)}')
    breakpoint()


def discretize(dim, n, order):
    # ### problem definition
    # see the dolfinx-tutorial by J. S. Dokken
    # https://jsdokken.com/dolfinx-tutorial/chapter2/nonlinpoisson.html#
    # for more information on nonlinear problems in dolfinx

    import dolfinx as df
    import numpy as np
    import ufl
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    if dim == 2:
        domain = df.mesh.create_unit_square(comm, n, n)
    elif dim == 3:
        domain = df.mesh.create_unit_cube(comm, n, n, n)
    else:
        raise NotImplementedError


    V = df.fem.functionspace(domain, ('Lagrange', order))

    # Dirichlet bc
    g = df.fem.Constant(domain, df.default_scalar_type(1.0))
    def dirichlet_boundary(x):
        return np.isclose(x[0], 1.0)

    c = df.fem.Constant(domain, df.default_scalar_type(1.0))
    def q(u):
        return 1 + c*u**2

    uh = df.fem.Function(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(domain)
    f = x[0] * ufl.sin(x[1])
    F = q(uh) * ufl.dot(ufl.grad(uh), ufl.grad(v)) * ufl.dx - f * v * ufl.dx

    # ### pyMOR wrapping
    from pymor.bindings.fenicsx import BCGeom, FenicsxOperator, FenicsxVectorSpace, FenicsxVisualizer
    from pymor.models.basic import StationaryModel
    from pymor.operators.constructions import VectorOperator

    def parameter_setter(mu):
        v = mu['c'].item()
        c.value = df.default_scalar_type(v)

    space = FenicsxVectorSpace(V)
    op = FenicsxOperator(F, space, space, uh, (BCGeom(g, dirichlet_boundary, V),),
                        parameter_setter=parameter_setter,
                        parameters={'c': 1},
                        solver_options={'inverse': {'type': 'newton', 'rtol': 1e-6}})
    rhs = VectorOperator(op.range.zeros())

    fom = StationaryModel(op, rhs,
                          visualizer=FenicsxVisualizer(space))

    return fom


if __name__ == '__main__':
    run(main)
