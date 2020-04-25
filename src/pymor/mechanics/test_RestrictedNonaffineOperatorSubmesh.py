#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""test RestrictedNonaffineOperatorSubmesh

Usage:
    test_RestrictedNonaffineOperatorSubmesh.py [options] NTEST

Arguments:
    NTEST               Size of the testing set.

Options:
    -h, --help          Show this message.
    --plot              Plot mesh and dofs.
"""

import sys
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
from docopt import docopt

from pymor.bindings.fenics import FenicsVectorSpace
from pymor.mechanics.operators import NonaffineOperator
from pymor.mechanics.creep import ParametricKelvinChain
from pymor.parameters.base import ParameterType
from pymor.parameters.spaces import CubicParameterSpace
from pymor.vectorarrays.numpy import NumpyVectorSpace


def parse_arguments(args):
    args = docopt(__doc__, args)
    args['NTEST'] = int(args['NTEST'])
    return args


def discretize(args):
    mesh = df.UnitSquareMesh(20, 20)
    V = df.VectorFunctionSpace(mesh, "CG", 1)
    v = df.TestFunction(V)

    class Omega_0(df.SubDomain):
        tol = 1e-14

        def inside(self, x, on_boundary):
            return x[1] <= 0.5 + self.tol

    class Omega_1(df.SubDomain):
        tol = 1e-14

        def inside(self, x, on_boundary):
            return x[1] >= 0.5 - self.tol

    materials = df.MeshFunction('size_t', mesh, mesh.geometric_dimension())
    subdomain_1 = Omega_1()

    materials.set_all(1)
    subdomain_1.mark(materials, 99)

    degree = 1
    metadata = {'quadrature_degree': degree, 'quad_scheme': 'default'}
    dx = df.Measure('dx')(metadata=metadata, subdomain_data=materials)

    pt = ParameterType({
        'D': (3,),
        'E0': (),
    })

    moduli = {}
    for m in range(3):
        moduli[f'D{m}'] = df.Constant(1.0)
    E0 = df.Constant(1.0)

    def mu_setter(mu):
        for i in range(3):
            moduli[f'D{i}'].assign(df.Constant(mu["D"][i]))
        E0.assign(df.Constant(mu["E0"]))
        return

    mat_param = {
        'nkelem': 3,
        'stiffness_moduli': tuple(moduli.values()),
        'tau': [3.0, 20.0, 40.0],
        'E0': E0,
        'dt': df.Constant(1.0),
        'parameter_setter': mu_setter,
        'parameter_type': pt
    }

    kc = ParametricKelvinChain(mesh, degree, parameters=mat_param)
    form = kc.get_nonaffine_form(v, dx(99))

    ps = CubicParameterSpace(pt, ranges={
        'D': (1e4, 3e4),
        'E0': (2e4, 3e4),
    })

    # TODO: if material is Parametric then I don't need to pass parameter_setter and
    # parameter_type as arguments to NonaffineOperator
    operator = NonaffineOperator(form, FenicsVectorSpace(V), FenicsVectorSpace(V),
                                 material=kc, dirichlet_bc=None, parameter_setter=mu_setter,
                                 parameter_type=pt, restriction_method='submesh')

    # select interpolation dofs in (x - 0.5)^2 + (y - 0.75)^2 < 0.2^2
    dofcoord = V.tabulate_dof_coordinates()
    dofs = np.where(dofcoord[:, 1] >= 0.5)[0]
    idofs = np.where((dofcoord[:, 0] - 0.5)**2
                     + (dofcoord[:, 1] - 0.75)**2 < 0.2**2)[0]

    if args['--plot']:
        df.plot(materials)
        df.plot(mesh)
        plt.plot(dofcoord[dofs, 0], dofcoord[dofs, 1], 'ro')
        plt.plot(dofcoord[idofs, 0], dofcoord[idofs, 1], 'bo')
        plt.show()

    op_r = operator.restricted(idofs)
    return (operator, idofs), op_r, ps


def main(args):
    args = parse_arguments(args)
    (operator, dofs), (restricted_operator, source_dofs), ps = discretize(args)
    restricted_range_dofs = restricted_operator.restricted_range_dofs

    # define some arbitrary U
    u = df.Function(operator.range.V)
    dim = operator.range.dim
    u.vector()[:] = np.random.rand(dim)

    u_r = df.interpolate(u, restricted_operator.op.range.V)  # interpolate u from CG space onto reduced CG space
    u_err = u.vector()[dofs] - u_r.vector()[restricted_range_dofs]
    assert np.allclose(u_err, 0.0)

    U = operator.source.make_array([u.vector()])  # FenicsVectorSpace
    U_r = restricted_operator.source.make_array(
        [u_r.vector()[:]])  # NumpyVectorSpace

    testing_set = ps.sample_randomly(args['NTEST'])

    test = 1
    for mu in testing_set:
        print("\n Solving for mu = ", mu)
        V = operator.apply(U, mu)  # FenicsVectorSpace
        V_r = restricted_operator.apply(U_r, mu)  # NumpyVectorSpace
        V_dofs = NumpyVectorSpace.make_array(V.dofs(dofs))
        ERR = V_dofs - V_r
        test *= np.isclose(ERR.l2_norm(), 0.0)

    if test:
        print("test RestrictedNonaffineOperatorSubmesh passed.")
    else:
        print("test RestrictedNonaffineOperatorSubmesh failed.")


if __name__ == "__main__":
    main(sys.argv[1:])
