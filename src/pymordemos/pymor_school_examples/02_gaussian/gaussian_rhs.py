#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test DEIM IQP in 2d by interpolating

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
    --err                   Print errors in max norm.
    --plot                  Plot meshes, integration points and singular
                            value decay.
    --product               Use inner product on L2(Ω) (mass). Without this option
                            the euclidean product/norm will be used.
    --rtol=rtol             Relative tolerance in POD. [default: 1e-8]
    --l2_err=err            Bound l2-approximation error by this value. [default: 0.0]
    --range=R               Range of parameter space given as string
                            containing a tuple of floats. [default: (-1.0, 1.0)]
"""
# TODO: maybe add option to use greedy instead of pod for deim basis generation

import sys
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import meshio
import ufl
import warnings

sys.path.append(r"/home/pdiercks/python-projects/helpers")
from table import print_table

# pymor
from pymor.basic import *
from pymor.core.logger import getLogger
from pymor.operators.my_ei import FenicsEmpiricalInterpolatedOperator
from pymor.bindings.fenics import FenicsVectorSpace
from pymor.bindings.my_fenics import FenicsMatrixOperator, FenicsNonaffineVectorOperator
from pymor.parameters.base import ParameterType
from pymor.tools.floatcmp import float_cmp_all

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

def perform_deim(parameter_space, operator, args, product=None):
    if operator.restriction_method == 'IQP':
        S = operator.quadrature_range.empty()
    else:
        S = operator.range.empty()
    training_set = parameter_space.sample_randomly(args['NSNAPSHOTS'])
    print("Solving on training set ... ")
    for mu in training_set:
        g = operator.as_range_array(mu)
        S.append(g)

    print("Performing DEIM algorithm ...")
    options = {}
    rtol = args['--rtol']
    options["l2_err"] = args['--l2_err']
    dofs, cb, data = deim(S, rtol=rtol, product=product, pod_options=options)
    return dofs, cb, data

def compute_errors(parameter_space, n_test, ufl_expr, operator, degree, product=None):
    ABS_ERR = list()
    REL_ERR = list()
    DEIM_BOUNDS = list()
    space = operator.range
    v = df.TestFunction(operator.range.V)
    method = operator.restricted_operator.op.restriction_method
    T = np.linalg.inv(operator.interpolation_matrix)
    U = operator.collateral_basis.to_numpy()
    metadata = {"quadrature_degree": degree, "quadrature_scheme": "default"}
    dx = df.dx(metadata=metadata)

    # ### stochastic testing
    for mu in parameter_space.sample_randomly(n_test):
        if method == 'submesh':
            fom = space.make_array([df.assemble(
                ufl_expr(mu) * v * dx)])# use dx everywhere
        elif method == 'IQP':
            vec = local_project(ufl_expr(mu), space.V).vector()
            fom = space.make_array([vec])
        else:
            assert False
        rom = operator.as_range_array(mu)
        # precomputations for error bound
        I = np.eye(fom.data.size)
        bvec = np.dot(I - np.dot(U.T, U), fom.to_numpy().T)
        b0 = np.linalg.norm(T, ord=2)
        if product:
            # use weighted norm
            aerr = np.sqrt(product.pairwise_apply2(fom - rom, fom - rom))[0]
            rerr = np.sqrt(aerr / product.pairwise_apply2(fom, fom))[0]
            # error bound only defined in 2-norm
            bound = b0# keep track of norm(inv(P.T U)) instead
        else:
            # use euclidean norm
            aerr = ((fom - rom).l2_norm())[0]
            rerr = ((fom - rom).l2_norm() / fom.l2_norm())[0]
            b1 = np.linalg.norm(bvec)
            bound = b0 * b1
        assert aerr <= bound
        ABS_ERR.append(aerr)
        REL_ERR.append(rerr)
        DEIM_BOUNDS.append(bound)
    return (ABS_ERR, REL_ERR, DEIM_BOUNDS)


def discretize_fenics(args):
    degree = args['QUAD_DEG']
    mesh = df.Mesh("data/gaussian.xml")
    subdomains = df.MeshFunction("size_t", mesh, "data/gaussian_physical_region.xml")
    boundaries = df.MeshFunction("size_t", mesh, "data/gaussian_facet_region.xml")
    x = ufl.SpatialCoordinate(mesh)


    # ### FE space
    V = df.FunctionSpace(mesh, "Lagrange", degree)
    v = df.TestFunction(V)
    u = df.TrialFunction(V)

    # ### Quadrature Space
    QE = df.FiniteElement("Quadrature", mesh.ufl_cell(), degree=degree, quad_scheme="default")
    Q = df.FunctionSpace(mesh, QE)
    psi = df.TrialFunction(Q)
    w = df.TestFunction(Q)
    q_points = Q.tabulate_dof_coordinates()
    if degree == 1:
        assert mesh.cells().shape[0] == q_points.shape[0]
    elif degree == 2:
        assert mesh.cells().shape[0]*3 == q_points.shape[0]
    else:
        raise NotImplementedError("QUAD_DEG should be 1 or 2.")

    metadata = {"quadrature_degree": degree, "quadrature_scheme": "default"}
    dx = df.dx(metadata=metadata)

    # ### products
    if args['--product']:
        mass = u * v * df.dx
        mass_iqp = psi * w * dx
        mass_matrix = df.assemble(mass)
        mass_matrix_iqp = df.assemble(mass_iqp)
        product = FenicsMatrixOperator(mass_matrix, V, V, name="mass")
        product_iqp = FenicsMatrixOperator(mass_matrix_iqp, Q, Q, name="mass")
    else:
        product = None
        product_iqp = None
    products = (product, product_iqp)

    # ### Gaussian
    m0 = df.Function(V)
    m1 = df.Function(V)
    m0.assign(df.Constant(1.0))
    m1.assign(df.Constant(1.0))
    gaussian_expr = ufl.exp(
            -2.0*(x[0] - m0)**2 - 2.0*(x[1] - m1)**2
        )
    def ufl_gaussian(mu):
        return ufl.exp(
            -2.0*(x[0] - mu["x"])**2 - 2.0*(x[1] - mu["y"])**2
        )
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

    # ### Linear form
    L = gaussian_expr * v * dx# use same integration rule everywhere

    # ### parametrization
    parameter_type = ParameterType({"x": (), "y": ()})
    parameter_space = CubicParameterSpace(
        parameter_type, ranges={"x": args['--range'],
                                "y": args['--range']})
    def param_setter(mu):
        m0.assign(df.Constant(float(mu["x"])))
        m1.assign(df.Constant(float(mu["y"])))

    # ### Operators
    range_space = FenicsVectorSpace(V)
    rhs_std = FenicsNonaffineVectorOperator(L, range_space,
                                            parameter_setter=param_setter,
                                            parameter_type=parameter_type,
                                            restriction_method='submesh')



    rhs_iqp = FenicsNonaffineVectorOperator(gaussian_expr,
                                            range_space,
                                            quadrature_space=FenicsVectorSpace(Q),
                                            parameter_setter=param_setter,
                                            parameter_type=parameter_type,
                                            restriction_method='IQP',
                                            numpy_form=numpy_gaussian)
    return rhs_std, rhs_iqp, parameter_space, products, ufl_gaussian

def main(args):
    args = parse_arguments(args)
    STD, IQP, pspace, products, g_ufl = discretize_fenics(args)
    # misc
    fe_space = STD.range
    Q = IQP.quadrature_range.V
    points = Q.tabulate_dof_coordinates()
    v = df.TestFunction(fe_space.V)

    # ### STANDARD DEIM
    idofs, basis, data = perform_deim(
        pspace, STD, args, product=products[0])
    #  EI_STD = EmpiricalInterpolatedOperator(STD, collateral_basis=basis,
    #                                         interpolation_dofs=idofs, triangular=False)
    EI_STD = FenicsEmpiricalInterpolatedOperator(STD, collateral_basis=basis,
                                                 interpolation_dofs=idofs, triangular=False)
    # ### IQP DEIM
    q_idofs, q_basis, q_data = perform_deim(
        pspace, IQP, args, product=products[1])

    EI_IQP = FenicsEmpiricalInterpolatedOperator(IQP, collateral_basis=q_basis,
                                                 interpolation_dofs=q_idofs, triangular=False)
    ESTIMATOR_IQP = FenicsEmpiricalInterpolatedOperator(IQP, collateral_basis=q_basis,
                                                        interpolation_dofs=q_idofs,
                                                        triangular=False, estimate_error=True)

    print("Testing randomly ...")
    results = compute_errors(pspace, args['NTEST'], g_ufl, EI_STD, args['QUAD_DEG'],
                             product=products[0])
    results_iqp = compute_errors(pspace, args['NTEST'], g_ufl, ESTIMATOR_IQP, args['QUAD_DEG'],
                                 product=products[1])

    if args['--err']:
        header = "Compare DEIM approximations to FOM FE Vector " + \
        f"using testing set of size {args['NTEST']}"
        cols = ['deim variant', 'max abs error', 'max rel error',
                'min err bound', 'idofs', 'total dofs']
        fmt = ['<', '^', '^', '^', '^', '^']
        width = [15, 25, 25, 25, 15, 15]
        rows = [
            ['standard', max(results[0]), max(results[1]), min(results[2]), idofs.size, EI_STD.range.V.dim()],
            ['IQP', max(results_iqp[0]), max(results_iqp[1]), min(results_iqp[2]), q_idofs.size, Q.dim()]
        ]
        print_table(header, cols, fmt, width, rows)

    # plotting
    if args['--plot']:
        mesh = EI_STD.range.V.mesh()
        submesh = EI_STD.restricted_operator.op.range.V.mesh()
        p = points[q_idofs]

        plt.figure(1)
        df.plot(mesh)
        plt.plot(points[:, 0], points[:, 1], 'k.')# all integration points
        plt.plot(p[:, 0], p[:, 1], 'ro')# points used for interpolation

        plt.figure(2)
        df.plot(submesh)
        plt.plot(p[:, 0], p[:, 1], 'r+')# points used for interpolation

        plt.figure(3)
        nvals = max([data["svals"].size, q_data["svals"].size])
        plt.semilogy(data["svals"] / data["svals"][0], 'r.-', label="std")
        plt.semilogy(q_data["svals"] / q_data["svals"][0], 'b.-', label="IQP")
        plt.semilogy(args['--rtol'] * np.ones(nvals), 'k-', label="rtol")
        plt.legend()

        plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
