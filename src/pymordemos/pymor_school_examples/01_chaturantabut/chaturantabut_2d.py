#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test DEIM IQP in 2d by interpolating

    s(x; µ) = 1 / sqrt(
        (x0 - µ0)**2 + (x1 - µ1)**2 + 0.1**2
    )
    for µ in P=range ** 2 over Ω =[0.1, 0.9]**2

empirically.

Source:
    Chaturantabut and Sorensen (2010): "Nonlinear model
    reduction via discrete empirical interpolation"
    Section 3.3.2

Usage:
    chaturantabut_2d.py [options] ORDER QUAD_DEG NTRAIN NTEST
    chaturantabut_2d.py -h | --help

Arguments:
    ORDER               Finite element order.
    QUAD_DEG            Quadrature Degree.
    NTRAIN              Size of training set.
    NTEST               Size of the testing set.

Options:
    --plot                  Plot meshes, integration points and singular
                            value decay and average error.
    --product               Use inner product on L2(Ω) (mass). Without this option
                            the euclidean product/norm will be used.
    --rtol=rtol             Relative tolerance in POD. [default: 1e-8]
    --l2_err=err            Bound l2-approximation error by this value. [default: 0.0]
    --range=R               Range of parameter space given as string
                            containing a tuple of floats. [default: (-1.0, -0.01)]
"""

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
from docopt import docopt
from pymor.basic import *
from pymor.core.logger import getLogger
from pymor.operators.my_ei import FenicsEmpiricalInterpolatedOperator
from pymor.bindings.fenics import FenicsVectorSpace, FenicsMatrixOperator
from pymor.bindings.my_fenics import FenicsNonaffineVectorOperator
from pymor.parameters.base import ParameterType
from pymor.tools.timing import Timer

from fenics_modules.helpers import local_project

from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)

df.parameters["form_compiler"]["representation"] = "quadrature"
WARNING = 30
df.set_log_level(WARNING)

# ### helpers
def parse_arguments(args):
    args = docopt(__doc__, args)
    args["QUAD_DEG"] = int(args["QUAD_DEG"])
    args["NTRAIN"] = int(args["NTRAIN"])
    args["NTEST"] = int(args["NTEST"])
    args['--range'] = eval(args['--range'])
    args["--rtol"] = float(args["--rtol"])
    args["--l2_err"] = float(args["--l2_err"])
    assert isinstance(args['--range'], tuple)
    assert all(isinstance(x, float) for x in args['--range'])
    if args['--l2_err'] > 0.0:
        assert args['--product'] is None
    return args

def perform_deim(parameter_space, operator, args, product=None):
    if operator.restriction_method == 'IQP':
        S = operator.quadrature_range.empty()
    else:
        S = operator.range.empty()
    training_set = parameter_space.sample_uniformly(args['NTRAIN'])
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

def compute_errors(args, parameter_space, ufl_expr, rhs, ei_rhs, product=None, variant=None):
    results = {
        "abs_err": [],
        "rel_err": [],
        "bounds": [],
        "PTU": [],
        "pod_err": [],
        "time": []
    }
    T = np.linalg.inv(ei_rhs.interpolation_matrix)
    W = ei_rhs.collateral_basis.to_numpy()

    # ### stochastic testing
    timer = Timer("timing deim variant" + variant)
    for mu in parameter_space.sample_uniformly(args["NTEST"]):
        U = rhs.as_range_array(mu)
        timer.start()
        URB = ei_rhs.as_range_array(mu)
        timer.stop()
        results["time"].append(timer.dt)
        # precomputations for error bound
        I = np.eye(U.data.size)
        bvec = np.dot(I - np.dot(W.T, W), U.to_numpy().T)
        b0 = np.linalg.norm(T, ord=2)
        if product:
            # use weighted norm
            aerr = np.sqrt(product.pairwise_apply2(U - URB, U - URB)[0])
            rerr = aerr / np.sqrt(product.pairwise_apply2(U, U)[0])
            # error bound only defined in 2-norm
            bound = b0# keep track of norm(inv(P.T U)) instead
            # TODO: how to compute a weighted norm of a matrix?
        else:
            # use euclidean norm
            aerr = (U - URB).l2_norm()[0]
            rerr = ((U - URB).l2_norm() / U.l2_norm())[0]
            b1 = np.linalg.norm(bvec)
            bound = b0 * b1
            assert aerr <= bound
        results["abs_err"].append(aerr)
        results["rel_err"].append(rerr)
        results["bounds"].append(bound)
        results["PTU"].append(b0)
        results["pod_err"].append(b1)
    return results

def discretize_fenics(args):
    degree = args['QUAD_DEG']
    mesh = df.Mesh("data/chaturantabut_2d.xml")
    subdomains = df.MeshFunction("size_t", mesh, "data/chaturantabut_2d_physical_region.xml")
    boundaries = df.MeshFunction("size_t", mesh, "data/chaturantabut_2d_facet_region.xml")
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

    # ### function s
    m0 = df.Function(V)
    m1 = df.Function(V)
    m0.assign(df.Constant(1.0))
    m1.assign(df.Constant(1.0))
    expr = 1 / ufl.sqrt(
        (x[0] - m0)**2 + (x[1] - m1)**2 + 0.1**2
    )
    def ufl_s(mu):
        return 1 / ufl.sqrt(
        (x[0] - mu["x"])**2 + (x[1] - mu["y"])**2 + 0.1**2
    )

    def numpy_s(points,  mu):
        assert points.ndim in (2,)
        x = points[:, 0]
        y = points[:, 1]
        return 1 / np.sqrt(
            (x - mu["x"])**2 + (y - mu["y"])**2 + 0.1**2
        )

    # ### Linear form
    L = expr * v * dx

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
    STD = FenicsNonaffineVectorOperator(L, range_space,
                                        parameter_setter=param_setter,
                                        parameter_type=parameter_type,
                                        restriction_method='submesh')



    IQP = FenicsNonaffineVectorOperator(expr,
                                        range_space,
                                        quadrature_space=FenicsVectorSpace(Q),
                                        parameter_setter=param_setter,
                                        parameter_type=parameter_type,
                                        restriction_method='IQP',
                                        numpy_form=numpy_s)
    return STD, IQP, parameter_space, products, ufl_s

def main(args):
    args = parse_arguments(args)
    STD, IQP, pspace, products, s_ufl = discretize_fenics(args)
    # misc
    #  fe_space = STD.range
    Q = IQP.quadrature_range.V
    points = Q.tabulate_dof_coordinates()
    #  v = df.TestFunction(fe_space.V)

    # ### ERROR ESTIMATION
    errors = {
        "standard": {
            "pod_err": [],
            "avg_err": [],
            "avg_bound": [],
            "approx_bound": [],
            "time": []
        },
        "iqp": {
            "pod_err": [],
            "avg_err": [],
            "avg_bound": [],
            "approx_bound": [],
            "time": []
        }
    }

    # ### STANDARD DEIM
    idofs, basis, data = perform_deim(pspace, STD, args, product=products[0])
    # ### IQP DEIM
    q_idofs, q_basis, q_data = perform_deim(pspace, IQP, args, product=products[1])

    print("Testing uniformly...")
    basis_sizes = [x * 5 for x in range(1, 8)]
    for i_N, N in enumerate(basis_sizes):
        EI_STD = EmpiricalInterpolatedOperator(STD, collateral_basis=basis[:N],
                                               interpolation_dofs=idofs[:N], triangular=False)
        EI_IQP = FenicsEmpiricalInterpolatedOperator(IQP, collateral_basis=q_basis[:N],
                                                     interpolation_dofs=q_idofs[:N], triangular=False)
        ESTIMATOR_IQP = FenicsEmpiricalInterpolatedOperator(IQP, collateral_basis=q_basis[:N],
                                                            interpolation_dofs=q_idofs[:N],
                                                            triangular=False, estimate_error=True)
        R = compute_errors(args, pspace, s_ufl, STD, EI_STD, variant="standard")
        RIQP = compute_errors(args, pspace, s_ufl, IQP, ESTIMATOR_IQP, variant="iqp")

        for variant, results in zip(["standard", "iqp"], [R, RIQP]):
            if variant == "standard":
                svals = data["all_svals"]
            elif variant == "iqp":
                svals = q_data["all_svals"]
            else:
                assert False
            errors[variant]["pod_err"].append(sum(results["pod_err"]) / args["NTEST"])
            errors[variant]["avg_err"].append(sum(results["abs_err"]) / args["NTEST"])
            errors[variant]["avg_bound"].append(sum(results["bounds"]) / args["NTEST"])
            errors[variant]["approx_bound"].append(sum(results["PTU"]) / args["NTEST"] * svals[N])
            errors[variant]["time"].append(sum(results["time"]) / args["NTEST"])

    mesh = EI_STD.range.V.mesh()
    submesh = EI_STD.restricted_operator.op.range.V.mesh()
    msh_smry = f"""Mesh data:
        cells in mesh:                      {mesh.num_cells()}
        cells in submesh:                   {submesh.num_cells()}
        quadrature points:                  {Q.dim()}
        interpolation quadrature points:    {q_idofs.size}\n"""
    print(msh_smry)

    # plotting
    if args['--plot']:
        p = points[q_idofs]

        plt.figure(1)
        plt.title("Mesh with quadrature points (black) and interpolation quadrature points (red).")
        df.plot(mesh)
        plt.plot(points[:, 0], points[:, 1], 'k.')# all integration points
        plt.plot(p[:, 0], p[:, 1], 'ro')# points used for interpolation

        plt.figure(2)
        plt.title("Submesh used in empricial interpolation with interpolation quadrature points for"
                  + "comparison")
        df.plot(submesh)
        plt.plot(p[:, 0], p[:, 1], 'r+')# points used for interpolation

        plt.figure(3)
        plt.title("Singular value decay for both DEIM variants")
        nvals = max([data["svals"].size, q_data["svals"].size])
        plt.semilogy(data["svals"] / data["svals"][0], 'r.-', label="std")
        plt.semilogy(q_data["svals"] / q_data["svals"][0], 'b.-', label="IQP")
        plt.semilogy(args['--rtol'] * np.ones(nvals), 'k-', label="rtol")
        plt.legend()

        plt.figure(4)
        plt.title("Average error and bounds for DEIM variants")
        c = ['r', 'b']
        for i_V, variant in enumerate(["standard", "iqp"]):
            plt.semilogy(basis_sizes, errors[variant]["pod_err"], c[i_V]+'^--', label="pod error")
            plt.semilogy(basis_sizes, errors[variant]["avg_err"], c[i_V]+'o-', label="avg error")
            plt.semilogy(basis_sizes, errors[variant]["avg_bound"], c[i_V]+'.-', label="avg error bound")
            plt.semilogy(basis_sizes, errors[variant]["approx_bound"], c[i_V]+'-*', label="approx error bound")
        plt.xlabel("number of basis functions used.")
        plt.legend()

        plt.figure(5)
        plt.title("CPU time for evaluation")
        for i_V, variant in enumerate(["standard", "iqp"]):
            plt.semilogy(basis_sizes, errors[variant]["time"], c[i_V]+'^--', label=variant)
        plt.xlabel("number of basis functions used.")
        plt.legend()

        plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
