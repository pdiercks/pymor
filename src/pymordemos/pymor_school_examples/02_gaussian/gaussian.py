#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test DEIM IQP in 2d with solving

    int_Ω grad(u) · grad(v) dx = int_Ω g(µ) v dx for all x in Ω = (-1, 1)²

and interpolating

    g(x; µ) = exp(-2(x[0]-µ[0])**2 - 2(x[1]-µ[1])**2)

empirically.

This example is taken from the RBniCS website
    https://rbnics.gitlab.io/RBniCS-jupyter/tutorial_gaussian_eim.html

Usage:
    gaussian.py [options] QUAD_DEG NSNAPSHOTS NTEST
    gaussian.py -h | --help

Arguments:
    QUAD_DEG            Quadrature Degree.
    NSNAPSHOTS          Number of deim snapshots.
    NTEST               Size of the testing set.

Options:
    --err                   Print errors in l2 norm.
    --plot                  Plot meshes, integration points and singular
                            value decay.
    --product               Use inner product on L2(Ω) (mass). Without this option
                            the euclidean product/norm will be used.
    --rtol=rtol             Relative tolerance in POD. [default: 1e-8]
    --l2_err=err            Bound l2-approximation error by this value. [default: 0.0]
    --range=R               Range of parameter space given as string
                            containing a tuple of floats. [default: (-1.0, 1.0)]
    --to-latex              Save results as latex table.
    --to-csv                Save data to csv, to be read via pgfplots.
    --to-pgf                Save plot to pgf, to include in latex.
"""

import sys
import dolfin as df
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ufl
import warnings
#  sys.path.append(r"/home/pdiercks/python-projects/helpers")
#  from table import print_table

# pymor
from pymor.algorithms.ei import deim
from pymor.models.basic import StationaryModel
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.operators.my_ei import FenicsEmpiricalInterpolatedOperator
from pymor.operators.constructions import LincombOperator
from pymor.bindings.fenics import FenicsVectorSpace, FenicsMatrixOperator
from pymor.bindings.my_fenics_old import FenicsNonaffineVectorOperator
from pymor.parameters.base import ParameterType
from pymor.parameters.spaces import CubicParameterSpace
from pymor.tools.timing import Timer

from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
from docopt import docopt
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

df.parameters["form_compiler"]["representation"] = "quadrature"
WARNING = 30
df.set_log_level(WARNING)

# ### helpers


def parse_arguments(args):
    args = docopt(__doc__, args)
    args["QUAD_DEG"] = int(args["QUAD_DEG"])
    args["NSNAPSHOTS"] = int(args["NSNAPSHOTS"])
    args["NTEST"] = int(args["NTEST"])
    args['--range'] = eval(args['--range'])
    args["--rtol"] = float(args["--rtol"])
    args["--l2_err"] = float(args["--l2_err"])
    assert isinstance(args['--range'], tuple)
    assert all(isinstance(x, float) for x in args['--range'])
    if args['--l2_err'] > 0.0:
        assert args['--product'] is None
    if args['--to-pgf']:
        assert args['--plot'], 'Cannot save to pgf without "--plot" option.'
    if args['--to-csv']:
        assert args['--err'], 'Cannot save to csv without "--err" option.'
    return args


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


def compute_errors(fom, rom, args, product=None):
    results = {}
    ABS_ERR = list()
    REL_ERR = list()
    FOM_TIME = list()
    TIME = list()

    # ### stochastic testing
    timer = Timer(f'{rom.name} solve')
    for mu in fom.parameter_space.sample_randomly(args['NTEST']):
        timer.start()
        U = fom.solve(mu)
        timer.stop()
        FOM_TIME.append(timer.dt)
        timer.start()
        URB = rom.solve(mu)
        timer.stop()
        TIME.append(timer.dt)
        if args['--product']:
            # use weighted norm
            aerr = fom.mass_norm(U - URB)[0]  # return float for print_table
            rerr = aerr / fom.mass_norm(U)[0]
        else:
            # use euclidean norm
            aerr = (U - URB).l2_norm()[0]
            rerr = ((U - URB).l2_norm() / U.l2_norm())[0]
        ABS_ERR.append(aerr)
        REL_ERR.append(rerr)
    results["abs err"] = ABS_ERR
    results["rel err"] = REL_ERR
    results["t"] = TIME
    results["t-fom"] = FOM_TIME
    return results


def discretize_fenics(args):
    mesh = df.Mesh("data/gaussian.xml")
    #  subdomains = df.MeshFunction("size_t", mesh, "data/gaussian_physical_region.xml")
    boundaries = df.MeshFunction("size_t", mesh, "data/gaussian_facet_region.xml")
    x = ufl.SpatialCoordinate(mesh)

    # ### parametrization
    parameter_type = ParameterType({"x": (), "y": ()})
    parameter_space = CubicParameterSpace(
        parameter_type, ranges={"x": args['--range'],
                                "y": args['--range']})

    # ### prepare FE spaces
    degree = args['QUAD_DEG']
    V = df.FunctionSpace(mesh, "Lagrange", degree)
    u = df.TrialFunction(V)
    v = df.TestFunction(V)
    metadata = {"quadrature_degree": degree, "quadrature_scheme": "default"}
    dx = df.dx(metadata=metadata)

    # ### quadrature space
    QE = df.FiniteElement("Quadrature", mesh.ufl_cell(), degree=degree, quad_scheme="default")
    Q = df.FunctionSpace(mesh, QE)

    print("\n *** MESH DATA ***\n")
    print("cells in mesh: ", mesh.num_cells())
    print("integration points: ", Q.dim())

    psi = df.TrialFunction(Q)
    w = df.TestFunction(Q)
    q_points = Q.tabulate_dof_coordinates()
    if degree == 1:
        assert mesh.cells().shape[0] == q_points.shape[0]
    elif degree == 2:
        assert mesh.cells().shape[0]*3 == q_points.shape[0]
    else:
        raise NotImplementedError("QUAD_DEG should be 1 or 2.")

    # ### gaussian
    m0 = df.Function(V)
    m1 = df.Function(V)
    m0.assign(df.Constant(1.0))
    m1.assign(df.Constant(1.0))
    gaussian_expr = ufl.exp(
        -2.0*(x[0] - m0)**2 - 2.0*(x[1] - m1)**2
    )

    # ### RHS
    L = gaussian_expr * v * dx

    def param_setter(mu):
        m0.assign(df.Constant(float(mu["x"])))
        m1.assign(df.Constant(float(mu["y"])))

    # ### LHS
    a = df.inner(df.grad(u), df.grad(v)) * dx
    mass = u * v * df.dx
    mass_iqp = psi * w * dx

    bcs = [
        df.DirichletBC(V, df.Constant(0.0), boundaries, 1),
        df.DirichletBC(V, df.Constant(0.0), boundaries, 2),
        df.DirichletBC(V, df.Constant(0.0), boundaries, 3)
    ]

    # ### Assembly
    matrix = df.assemble(a)
    mass_matrix = df.assemble(mass)
    mass_matrix_iqp = df.assemble(mass_iqp)
    mat0 = matrix.copy()
    mat0.zero()
    vector = df.assemble(L)
    for bc in bcs:
        bc.apply(mat0)
        bc.apply(vector)
        bc.zero(matrix)

    # ### Operators
    op = LincombOperator([FenicsMatrixOperator(mat0, V, V),
                          FenicsMatrixOperator(matrix, V, V)],
                         [1., 1.])
    product = FenicsMatrixOperator(mass_matrix, V, V, name="mass")
    product_iqp = FenicsMatrixOperator(mass_matrix_iqp, Q, Q, name="mass")
    fe_space = FenicsVectorSpace(V)
    rhs_std = FenicsNonaffineVectorOperator(L, fe_space,
                                            parameter_setter=param_setter,
                                            parameter_type=parameter_type,
                                            restriction_method='submesh')

    def numpy_gaussian(points,  mu):
        x = points[:, 0]
        y = points[:, 1]
        try:
            z = points[:, 2]
        except IndexError:
            z = 0
        return np.exp(
            -2.0*(x - mu["x"])**2 - 2.0*(y - mu["y"])**2
        )
    rhs_iqp = FenicsNonaffineVectorOperator(gaussian_expr,
                                            fe_space,
                                            quadrature_space=FenicsVectorSpace(Q),
                                            parameter_setter=param_setter,
                                            parameter_type=parameter_type,
                                            restriction_method='IQP',
                                            numpy_form=numpy_gaussian)
    fom_std = StationaryModel(op, rhs_std, products={"mass": product}, parameter_space=parameter_space)
    fom_iqp = StationaryModel(op, rhs_iqp, products={"mass": product_iqp}, parameter_space=parameter_space)
    return fom_std, fom_iqp


def perform_deim(model, args):
    """deim basis generation"""
    if model.rhs.restriction_method == 'IQP':
        D = model.rhs.quadrature_range.empty()
    else:
        D = model.rhs.range.empty()
    training_set = model.parameter_space.sample_randomly(args['NSNAPSHOTS'])
    print("Solving on training set ...")
    for mu in training_set:
        D.append(model.rhs.as_range_array(mu))

    print("Performing DEIM algorithm ...")
    options = {}
    rtol = args['--rtol']
    options["l2_err"] = args['--l2_err']
    dofs, cb, data = deim(D, rtol=rtol, product=model.mass_product, pod_options=options)
    return dofs, cb, data


def main(args):
    args = parse_arguments(args)
    fom_std, fom_iqp = discretize_fenics(args)
    idofs, basis, deim_data = perform_deim(fom_std, args)
    q_idofs, q_basis, q_deim_data = perform_deim(fom_iqp, args)
    ei_rhs = EmpiricalInterpolatedOperator(fom_std.rhs, collateral_basis=basis,
                                           interpolation_dofs=idofs, triangular=False)
    fom_std_ei = fom_std.with_(rhs=ei_rhs)

    print("\n *** affected cells ***\n")
    print(fom_std_ei.rhs.restricted_operator.op.range.V.mesh().num_cells())

    ei_iqp_rhs = FenicsEmpiricalInterpolatedOperator(fom_iqp.rhs,
                                                     interpolation_dofs=q_idofs,
                                                     collateral_basis=q_basis,
                                                     triangular=False)
    fom_iqp_ei = fom_iqp.with_(rhs=ei_iqp_rhs)

    with Timer('standard error estimation') as timer:
        RSTD = compute_errors(fom_std, fom_std_ei, args)
    with Timer('IQP error estimation') as timer:
        RIQP = compute_errors(fom_std, fom_iqp_ei, args)

    if args['--err']:
        # build DataFrames from both results dictionaries
        mesh = fom_std.operator.source.V.mesh()
        std = pd.DataFrame(data=RSTD)
        iqp = pd.DataFrame(data=RIQP)
        # process data and create final DataFrame
        d = {
            'deim variant': ['standard', 'IPI'],
            'max abs err': [
                np.amax(std["abs err"].values),
                np.amax(iqp["abs err"].values)
            ],
            'max rel err': [
                np.amax(std["rel err"].values),
                np.amax(iqp["rel err"].values)
            ],
            'interpolation dofs': [idofs.size, q_idofs.size],
            'total dofs': [fom_std.operator.source.V.dim(),
                           fom_iqp.rhs.quadrature_range.V.dim()],
        }
        errs = pd.DataFrame(data=d)
        print("\n*** error estimation ***\n")
        print(errs)

        # ### timing
        # mean time per solve
        t = {
            'mesh size': [mesh.num_cells(), ],
            'FOM': [std['t-fom'].mean(), ],
            'DEIM-Std': [std['t'].mean(), ],
            'DEIM-IQP': [iqp['t'].mean(), ],
        }
        timing = pd.DataFrame(data=t)
        print("\n*** mean time per solve ***\n")
        print(timing)

        if args['--to-latex']:
            path = '/home/pdiercks/tex-documents/notes/reduction/tables/'
            errs.to_latex(buf=path + 'gaussian.tex')
            timing.to_latex(buf=path + 'gaussian_cpu.tex')

    mesh = fom_std.operator.range.V.mesh()
    submesh = ei_rhs.restricted_operator.op.range.V.mesh()
    qp = fom_iqp.rhs.quadrature_range.V.tabulate_dof_coordinates()
    p = qp[q_idofs]
    # plotting
    if args['--plot']:
        plt.figure(1)
        df.plot(mesh)
        plt.plot(qp[:, 0], qp[:, 1], 'k.', markersize=1)  # all integration points
        plt.plot(p[:, 0], p[:, 1], 'ro', markersize=3, fillstyle='none')  # points used for interpolation
        if args['--to-pgf']:
            path = '/home/pdiercks/tex-documents/notes/reduction/tikz/'
            plt.savefig(path + 'GaussianMesh.pgf')

        plt.figure(2)
        df.plot(submesh)
        plt.plot(p[:, 0], p[:, 1], 'ro', markersize=3, fillstyle='none')  # points used for interpolation
        if args['--to-pgf']:
            path = '/home/pdiercks/tex-documents/notes/reduction/tikz/'
            plt.savefig(path + 'GaussianSubMesh.pgf')

        plt.figure(3)
        plt.semilogy(deim_data["svals"] / deim_data["svals"][0], 'r.-', label="std")
        plt.semilogy(q_deim_data["svals"] / q_deim_data["svals"][0], 'b.-', label="IQP")
        plt.legend()

        plt.show()

    if args['--to-csv']:
        svals = {
            'DEIM-Std': deim_data['svals'] / deim_data['svals'][0],
            'DEIM-IQP': q_deim_data['svals'] / q_deim_data['svals'][0],
        }
        s = pd.DataFrame(data={
            k: pd.Series(x) for k, x in svals.items()
        })
        path = '/home/pdiercks/tex-documents/notes/reduction/data/'
        s.to_csv(path + 'gaussian_svals.csv')


if __name__ == "__main__":
    main(sys.argv[1:])
