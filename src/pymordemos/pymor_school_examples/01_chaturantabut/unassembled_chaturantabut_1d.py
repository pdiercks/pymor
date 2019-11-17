#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test DEIM IQP in 1d with computing an approximation of

    s(x; µ) = (1-x) * cos(3*pi*µ*(1+x)) * exp(-(x+1)*µ)

Source:
    Chaturantabut and Sorensen (2010): "Nonlinear model
    reduction via discrete empirical interpolation"
    Section 3.3.1

Usage:
    truss.py [options] NELEMENTS ORDER NTRAIN NTEST
    truss.py -h | --help
    kernprof -l -v truss.py (profiling)

Arguments:
    NELEMENTS           Number of elements in df.UnitIntervalMesh.
    ORDER               Finite element order.
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
import ufl

# pymor
from docopt import docopt
from pymor.algorithms.ei import deim, ei_greedy
from pymor.algorithms.pod import pod
from pymor.bindings.fenics import FenicsVectorSpace, FenicsMatrixOperator
from pymor.bindings.mechanics import MechanicsOperator, MechanicsEmpiricalInterpolatedOperator
from pymor.parameters.base import ParameterType
from pymor.parameters.spaces import CubicParameterSpace

WARNING = 30
df.set_log_level(WARNING)

# ### helpers
def parse_arguments(args):
    args = docopt(__doc__, args)
    args["NELEMENTS"] = int(args["NELEMENTS"])
    args["ORDER"] = int(args["ORDER"])
    args["NTRAIN"] = int(args["NTRAIN"])
    args["NTEST"] = int(args["NTEST"])
    args["--rtol"] = float(args["--rtol"])
    return args

def discretize_fenics(args):
    degree = args['ORDER']
    mesh = df.IntervalMesh(args['NELEMENTS'], -1.0, 1.0)
    x = ufl.SpatialCoordinate(mesh)

    # ### FE space
    V = df.FunctionSpace(mesh, "Lagrange", degree)
    v = df.TestFunction(V)
    u = df.TrialFunction(V)

    # ### DG space
    DG = df.FunctionSpace(mesh, "DG", degree)
    d_test = df.TestFunction(DG)

    # ### function s
    m0 = df.Function(V)
    m0.assign(df.Constant(1.0))
    expr = (1 - x[0]) * ufl.cos(3 * df.DOLFIN_PI * m0 * (x[0] + 1)) * ufl.exp(-(x[0] + 1) * m0)

    # ### Linear form
    L = expr * v * df.dx
    L_dg = expr * d_test * df.dx

    # ### parametrization
    parameter_type = ParameterType({"mu": ()})
    parameter_space = CubicParameterSpace(
        parameter_type, ranges={"mu": (1.0, df.DOLFIN_PI)})

    def param_setter(mu):
        m0.assign(df.Constant(float(mu["mu"])))

    # ### Operators
    STD = MechanicsOperator(None, L, FenicsVectorSpace(V),
                            FenicsVectorSpace(V),
                            parameter_setter=param_setter,
                            parameter_type=parameter_type,
                            restriction_method='assemble_local')

    space = FenicsVectorSpace(DG)
    UDEIM = MechanicsOperator(None, L_dg, space, space,
                              parameter_setter=param_setter,
                              parameter_type=parameter_type,
                              restriction_method='assemble_local')
    return STD, UDEIM, parameter_space


def main(args):
    args = parse_arguments(args)
    std, udeim, ps = discretize_fenics(args)

    A = udeim.range.empty()
    B = std.range.empty()

    training_set = ps.sample_uniformly(args['NTRAIN'])
    print("Solving on training set ... ")
    U_dg = udeim.source.random()
    U_cg = std.source.random()

    for mu in training_set:
        A.append(udeim.apply(U_dg, mu))
        B.append(std.apply(U_cg, mu))

    print("Performing DEIM algorithm ...")
    options = {}
    rtol = args['--rtol']
    pod_options = {
        'return_evals': True,
        'return_svecs': True
    }
    idofs, cb, data = deim(A, rtol=rtol, pod_options=pod_options)
    basis = B.lincomb(data['svecs'])

    EI = MechanicsEmpiricalInterpolatedOperator(udeim, idofs, cb, False,
                                             assembled_basis=basis)
    mu = training_set[0]
    # ### mesh data
    cg_idofs, cg_cb, cd_data = deim(B, rtol=rtol)
    EI_STD = MechanicsEmpiricalInterpolatedOperator(std, cg_idofs, cg_cb, triangular=False)
    print(len(EI.restricted_operator.cells))
    print(len(EI_STD.restricted_operator.cells))

    V = std.source.V
    DG = udeim.source.V

    U = std.source.random()
    u = df.Function(V)
    u.vector().set_local(U._list[0].impl)
    u_proj = df.project(u, DG)
    U_proj = udeim.source.make_array([u_proj.vector()])
    g_exact = std.apply(U, mu)
    g_dg = EI.apply(U_proj, mu)

    plt.plot(g_dg.to_numpy().flatten(), 'b-o')
    plt.plot(g_exact.to_numpy().flatten(), 'r.-')
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1:])
