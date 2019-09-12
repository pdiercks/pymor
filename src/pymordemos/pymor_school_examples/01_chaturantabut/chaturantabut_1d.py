#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test DEIM IQP in 1d with computing an approximation of

    s(x; µ) = (1-x) * cos(3*pi*µ*(1+x)) * exp(-(x+1)*µ)

Source:
    Chaturantabut and Sorensen (2010): "Nonlinear model
    reduction via discrete empirical interpolation"
    Section 3.3.1

Usage:
    truss.py [options] NELEMENTS ORDER QUAD_DEG NTRAIN NTEST
    truss.py -h | --help
    kernprof -l -v truss.py (profiling)

Arguments:
    NELEMENTS           Number of elements in df.UnitIntervalMesh.
    ORDER               Finite element order.
    QUAD_DEG            Quadrature Degree.
    NTRAIN              Size of training set.
    NTEST               Size of testing set.

Options:
    --plot                  Plot singular value decay and average errors.
    --rtol=rtol             Relative tolerance in POD. [default: 1e-8]
    --l2_err=err            Bound l2-approximation error by this value. [default: 0.0]
"""

import sys
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import meshio
import ufl
import warnings

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
    args["NELEMENTS"] = int(args["NELEMENTS"])
    args["ORDER"] = int(args["ORDER"])
    args["QUAD_DEG"] = int(args["QUAD_DEG"])
    args["NTRAIN"] = int(args["NTRAIN"])
    args["NTEST"] = int(args["NTEST"])
    args["--rtol"] = float(args["--rtol"])
    args["--l2_err"] = float(args["--l2_err"])
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

def compute_errors(args, parameter_space, ufl_expr, rhs, ei_rhs):
    results = {
        "abs_err": [],
        "rel_err": [],
        "bounds": [],
        "PTU": [],
        "pod_err": []
    }
    T = np.linalg.inv(ei_rhs.interpolation_matrix)
    W = ei_rhs.collateral_basis.to_numpy()

    # ### stochastic testing
    for mu in parameter_space.sample_uniformly(args["NTEST"]):
        U = rhs.as_range_array(mu)
        URB = ei_rhs.as_range_array(mu)
        # precomputations for error bound
        I = np.eye(U.data.size)
        bvec = np.dot(I - np.dot(W.T, W), U.to_numpy().T)
        b0 = np.linalg.norm(T, ord=2)
        b1 = np.linalg.norm(bvec)
        bound = b0 * b1
        # use euclidean norm
        aerr = (U - URB).l2_norm()[0]
        rerr = ((U - URB).l2_norm() / U.l2_norm())[0]
        assert aerr <= bound
        results["abs_err"].append(aerr)
        results["rel_err"].append(rerr)
        results["bounds"].append(bound)
        results["PTU"].append(b0)
        results["pod_err"].append(b1)
    return results

def discretize_fenics(args):
    degree = args['QUAD_DEG']
    mesh = df.IntervalMesh(args['NELEMENTS'], -1.0, 1.0)
    x = ufl.SpatialCoordinate(mesh)

    # ### FE space
    V = df.FunctionSpace(mesh, "Lagrange", degree)
    v = df.TestFunction(V)
    u = df.TrialFunction(V)

    # ### Quadrature Space
    QE = df.FiniteElement("Quadrature", mesh.ufl_cell(), degree=degree, quad_scheme="default")
    Q = df.FunctionSpace(mesh, QE)
    q_points = Q.tabulate_dof_coordinates()
    if degree == 1:
        assert mesh.cells().shape[0] == q_points.shape[0]
    elif degree == 2:
        assert mesh.cells().shape[0]*2 == q_points.shape[0]
    else:
        raise NotImplementedError("QUAD_DEG should be 1 or 2.")

    metadata = {"quadrature_degree": degree, "quadrature_scheme": "default"}
    dx = df.dx(metadata=metadata)

    # ### function s
    m0 = df.Function(V)
    m0.assign(df.Constant(1.0))
    expr = (1 - x[0]) * ufl.cos(3 * df.DOLFIN_PI * m0 * (x[0] + 1)) * ufl.exp(-(x[0] + 1) * m0)

    def ufl_s(mu):
        return (1 - x[0]) * ufl.cos(3 * df.DOLFIN_PI * mu["mu"] * (x[0] + 1)) * ufl.exp(-(x[0] + 1) * mu["mu"])

    def numpy_s(points,  mu):
        assert points.ndim in (2,)
        x = points[:, 0]
        return (1 - x) * np.cos(3 * df.DOLFIN_PI * mu["mu"] * (x + 1)) * np.exp(-(x + 1) * mu["mu"])

    # ### Linear form
    L = expr * v * dx

    # ### parametrization
    parameter_type = ParameterType({"mu": ()})
    parameter_space = CubicParameterSpace(
        parameter_type, ranges={"mu": (1.0, df.DOLFIN_PI)})

    def param_setter(mu):
        m0.assign(df.Constant(float(mu["mu"])))

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
    Q = IQP.quadrature_range.V
    points = Q.tabulate_dof_coordinates()

    # ### ERROR ESTIMATION
    errors = {
        "standard": {
            "pod_err": [],
            "avg_err": [],
            "avg_bound": [],
            "approx_bound": []
        },
        "iqp": {
            "pod_err": [],
            "avg_err": [],
            "avg_bound": [],
            "approx_bound": []
        }
    }

    # ### STANDARD DEIM
    idofs, basis, data = perform_deim(pspace, STD, args, product=None)
    # ### IQP DEIM
    q_idofs, q_basis, q_data = perform_deim(pspace, IQP, args, product=None)

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
        R = compute_errors(args, pspace, s_ufl, STD, EI_STD)
        RIQP = compute_errors(args, pspace, s_ufl, IQP, ESTIMATOR_IQP)

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
        plt.title("Average error and bounds for DEIM variants")
        c = ['r', 'b']
        for i_V, variant in enumerate(["standard", "iqp"]):
            plt.semilogy(basis_sizes, errors[variant]["pod_err"], c[i_V]+'^--', label="pod error")
            plt.semilogy(basis_sizes, errors[variant]["avg_err"], c[i_V]+'o-', label="avg error")
            plt.semilogy(basis_sizes, errors[variant]["avg_bound"], c[i_V]+'.-', label="avg error bound")
            plt.semilogy(basis_sizes, errors[variant]["approx_bound"], c[i_V]+'-*', label="approx error bound")
        plt.xlabel("number of basis functions used.")
        plt.legend()

        plt.figure(2)
        plt.title("Singular value decay for both DEIM variants")
        nvals = max([data["all_svals"].size, q_data["all_svals"].size])
        plt.semilogy(data["all_svals"] / data["all_svals"][0], 'r.-', label="std")
        plt.semilogy(q_data["all_svals"] / q_data["all_svals"][0], 'b.-', label="IQP")
        plt.semilogy(args['--rtol'] * np.ones(nvals), 'k-', label="rtol")
        plt.legend()

        plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
