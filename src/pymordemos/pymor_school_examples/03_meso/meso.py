#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Simple tension of a 2d meso structure in Ω=(0, 1)²

    inclusion:  linear elastic material
    matrix:     viscoelastic material

Usage:
    meso.py [options] QUAD_DEG NTRAIN NTEST

Arguments:
    QUAD_DEG        Quadrature Degree.
    NTRAIN          Size of the training set.
    NTEST           Size of the testing set.

Options:
    -h, --help      Show this message.
    --plot-mesh     Plot mesh, to see which RVE type is used.
    --save          Save displacement trajectories for U and URB
                    for a particular mu to file.
    --err           Perform reduction error analysis.
"""

import sys
import time
import dolfin as df
import numpy as np
import matplotlib.pyplot as plt
import warnings

from math import sqrt
from ffc.quadrature.deprecation import QuadratureRepresentationDeprecationWarning
from docopt import docopt

from fenics_modules.io import MeshReader
from fenics_modules.meso import MesoStructure2D
from fenics_modules.linear_elasticity import ParametricLinearElasticMaterial
from fenics_modules.creep import ParametricKelvinChain

# pymor
from pymor.algorithms.ei import ei_greedy, deim
from pymor.algorithms.error import reduction_error_analysis
from pymor.algorithms.pod import pod
from pymor.bindings.fenics import (
    FenicsMatrixOperator,
    FenicsVectorSpace,
)
from pymor.bindings.my_fenics import FenicsOperator, FenicsVisualizer
from pymor.models.my_basic import InstationaryModel
from pymor.operators.constructions import (
    VectorOperator, LincombOperator, InverseOperator, Concatenation
)
from pymor.operators.ei import EmpiricalInterpolatedOperator
from pymor.parameters.base import ParameterType
from pymor.parameters.functionals import GenericParameterFunctional, ExpressionParameterFunctional
from pymor.parameters.spaces import CubicParameterSpace
from pymor.reductors.my_basic import InstationaryRBReductor
from pymor.reductors.coercive import CoerciveRBReductor
from pymor.vectorarrays.list import ListVectorArray

warnings.simplefilter("ignore", QuadratureRepresentationDeprecationWarning)
WARNING = 30
df.set_log_level(WARNING)

def parse_arguments(args):
    args = docopt(__doc__, args)
    args['QUAD_DEG'] = int(args['QUAD_DEG'])
    args['NTRAIN'] = int(args['NTRAIN'])
    args['NTEST'] = int(args['NTEST'])
    args['--save'] = int(args['--save'])
    return args

def discretize(args):
    mr = MeshReader('rve', path='./data/')
    data = mr.read(subdomains=True, boundaries=True)
    mesh = data['mesh']
    subdomains = data['subdomains']
    boundaries = data['boundaries']
    V = df.VectorFunctionSpace(mesh, "Lagrange", 1)
    if args['--plot-mesh']:
        plt.figure(1001)
        p = df.plot(subdomains)
        plt.colorbar(p)
        df.plot(mesh)
        plt.title('mesh and subdomains')
        plt.show()

    # ### bcs
    # see /home/pdiercks/python-projects/mesh/pygmsh/rve/rve.msh for marking
    bcs = [
        df.DirichletBC(V.sub(1), df.Constant(0), boundaries, 3),# bottom
        df.DirichletBC(V.sub(0), df.Constant(0), boundaries, 6)# left
    ]

    # ### load
    force = df.Expression(("0.0", "F"), degree=0, F=10.0, name='f')

    # ### linear elastic inclusions
    inclusion = ParametricLinearElasticMaterial(2, plane_stress=True, E=3e4, NU=0.3)

    # ### non-aging Kelvin chain
    NMOD = 3 # set number of stiffness moduli
    material_parameters = {
        "E_0": 3e4,
        "num_kelvin_elements": NMOD,
        "dt": 1.0,
        "plane_stress": True,
    }
    KC = ParametricKelvinChain(mesh, args['QUAD_DEG'], **material_parameters)
    moduli = {}
    for m in range(NMOD):
        moduli[f'D{m}'] = df.Constant(1.0)
    KC.stiffness_moduli = tuple(moduli.values())
    KC.tau = [3, 20, 40]

    NINC = 1
    meso = MesoStructure2D(V, subdomains=subdomains, boundaries=boundaries,
                           ninc=NINC, inclusion=inclusion, matrix=KC, QUAD_DEG=args['QUAD_DEG'],
                           subdomain_id={'matrix': 1, 'inclusion': 2})
    # ### Assembly
    l2_mat = meso.assemble_inner_product(name='l2')
    inclusion_mats = meso.assemble_inclusions()
    meso_matrix = meso.assemble_matrix()
    bc_mat = l2_mat.copy()
    bc_mat.zero()
    rhs = meso.assemble_rhs(force, 5)# 5 = pull at top

    # ### apply bcs
    for bc in bcs:
        for k in inclusion_mats.keys():
            bc.zero(inclusion_mats[k])
        bc.zero(meso_matrix)
        bc.zero(l2_mat)
        bc.apply(bc_mat)

    # ### parametrization
    # in general mu['component'][coordinate]
    # here mu['E'] is an array of shape (NINC,)
    # mu['E'][i] will give E for the i-th inclusion
    pt = ParameterType({'D': (NMOD,),
                        'E': (NINC,),
                        'NU': (NINC,),
                        'theta': ()})
    ps = CubicParameterSpace(pt, ranges={'D': (1e4, 3e4),
                                         'E': (2 * meso.matrix.E_0, meso.matrix.E_0 * 4),
                                         'NU': (0.2, 0.4),
                                         'theta': (-1.0, 1.0)})

    # ### create dict of parameter functionals for matrix operators
    names = [f'inc{i}_lambda' for i in range(NINC)] + \
            [f'inc{i}_mu' for i in range(NINC)] + \
            ['matrix'] + ['bc_mat']
    pf = dict.fromkeys(names)
    op = dict.fromkeys(names)
    for coordinate in range(NINC):
        pf[f'inc{coordinate}_lambda'] = ExpressionParameterFunctional(
            f"E[{coordinate}] * NU[{coordinate}] / ((1+NU[{coordinate}]) * (1-2*NU[{coordinate}]))",
            pt, name=f'inc{coordinate}_lambda')
        pf[f'inc{coordinate}_mu'] = ExpressionParameterFunctional(
            f'E[{coordinate}] / 2 / (1+NU[{coordinate}])', pt, name=f'inc{coordinate}_mu')
        op[f'inc{coordinate}_lambda'] = FenicsMatrixOperator(
            inclusion_mats[f'inc{coordinate}_lambda'], V, V, name=f'inc{coordinate}_lambda')
        op[f'inc{coordinate}_mu'] = FenicsMatrixOperator(
            inclusion_mats[f'inc{coordinate}_mu'], V, V, name=f'inc{coordinate}_mu')

    def effective_stiffness(mu):
        lmbd = meso.matrix.get_lambda()
        chain = 0
        for i in range(NMOD):
            chain += (1.0 - lmbd[i]) / mu["D"][i]
        compliance = 1 / meso.matrix.E_0 + chain
        return 1 / compliance

    pf_EBAR = GenericParameterFunctional(effective_stiffness, pt, name='EBAR')
    pf['matrix'] = pf_EBAR

    pf['bc_mat'] = 1.0
    op['matrix'] = FenicsMatrixOperator(meso_matrix, V, V, name='matrix')
    op['bc_mat'] = FenicsMatrixOperator(bc_mat, V, V, name='bc_mat')
    parameter_functionals = list(pf.values())
    operators = list(op.values())

    pf_load = GenericParameterFunctional(lambda mu: mu['theta'], pt, name='load')

    # ### pymor Operators
    l2_product = FenicsMatrixOperator(l2_mat, V, V, name='l2')
    operator = LincombOperator(operators, parameter_functionals)
    rhs_0 = LincombOperator([VectorOperator(FenicsVectorSpace(V).make_array([rhs]))],
                          [pf_load])


    # ### RHS
    def parameter_setter(mu):
        for i in range(NMOD):
            moduli[f'D{i}'].assign(
                df.Constant(mu["D"][i])
            )
    rhs_1 = FenicsOperator(meso.get_nonaffine_form(),
                           FenicsVectorSpace(V),
                           FenicsVectorSpace(V),
                           df.Function(V), # dummy
                           dirichlet_bc=bcs,
                           parameter_setter=parameter_setter,
                           parameter_type=pt,
                           restriction_method='assemble_local',
                           material=meso.matrix,
                           subdomain_data=subdomains)


    # initial_parameters = parameter functionals to compute U0
    initial_parameters = pf.copy()
    initial_parameters['matrix'] = meso.matrix.E_0
    A0 = operator.with_(coefficients=tuple(initial_parameters.values()))
    #  A0 = operator.assemble(initial_parameters)
    # have parameter functionals which are not known beforehand (E, NU) for the inclusion
    # and pf_EBAR which I would like to prescribe for the first time step.

    initial_data = Concatenation(
        [InverseOperator(A0), rhs_0]
    )

    t_end = 20
    visualizer = FenicsVisualizer(FenicsVectorSpace(V))
    fom = InstationaryModel(t_end, initial_data, operator, rhs_0, rhs_1,
                            parameter_space=ps, products={'l2': l2_product},
                            visualizer=visualizer)
    summary = f'''FEniCS model:
   number of elements:    {mesh.num_cells()}
   number of dofs:        {V.dim()}
   finite element order:  {V.ufl_element().degree()}
'''
    return fom, summary

def generate_rom(args, fom):
    """ROM generation with POD/DEIM"""
    snapshots = {}
    displacement_trajectories = {}
    force_trajectories = {}
    tic = time.time()

    training_set = fom.parameter_space.sample_randomly(args['NTRAIN'])
    U = fom.operator.range.empty()
    F = fom.operator.range.empty()

    print("Solving on training set ...")
    for (i, mu) in enumerate(training_set):
        UU, data = fom.solve(mu=mu, return_rhs=True)
        displacement_trajectories[f"U{i}"] = UU
        force_trajectories[f"F{i}"] = data['rhs']
        U.append(UU)
        F.append(data['rhs'])
    snapshots['displacement'] = displacement_trajectories
    snapshots['internal_forces'] = force_trajectories

    print("Performing POD based DEIM ...")
    idofs, cb, deim_data = deim(F, rtol=1e-8, product=fom.l2_product)
    ei_rhs = EmpiricalInterpolatedOperator(fom.rhs_1, collateral_basis=cb,
                                           interpolation_dofs=idofs, triangular=False)
    fom_ei = fom.with_(rhs_1=ei_rhs)

    print("Performing POD ...")
    pod_basis, pod_svals, pod_all_svals = pod(U, rtol=1e-8, product=fom.l2_product)
    rom_data = {"po_basis": pod_basis,
                "pod_svals": pod_svals,
                "idofs": idofs,
                "deim_basis": cb,
                "deim_data": deim_data,
               }

    # this estimator is just a dummy since I don't know the lower bound yet
    # for use of POD this should not have any effect, right?
    coercivity_estimator = ExpressionParameterFunctional('min(E)', fom.parameter_type)
    reductor = InstationaryRBReductor(fom_ei, product=fom_ei.l2_product,
                                      initial_data_product=fom_ei.l2_product)

    print("Reducing ...")
    reductor.extend_basis(pod_basis, method="trivial")
    rom = reductor.reduce()

    elapsed_time = time.time() - tic

    # generate summary
    real_rb_size = rom.solution_space.dim
    training_set_size = len(training_set)
    trajectories = len(UU)
    summary = f'''POD basis generation:
   size of training set:   {training_set_size}x{trajectories}
   actual basis size:      {real_rb_size}
   elapsed time:           {elapsed_time}
'''
    return rom, reductor, summary, snapshots, rom_data


def main(args):
    args = parse_arguments(args)
    fom, fom_summary = discretize(args)
    rom, reductor, red_summary, snapshots, rom_data = generate_rom(args, fom)

    if args['--err']:
        print('\nSearching for maximum error on random snapshots ...')
        results = reduction_error_analysis(rom,
                                       fom=fom,
                                       reductor=reductor,
                                       estimator=False,
                                       error_norms=(fom.l2_norm,),
                                       condition=False,
                                       test_mus=args['NTEST'],
                                       basis_sizes=1,# 0 will fail
                                       plot=True,
                                       random_seed=999)
        print('\n*** RESULTS ***\n')
        print(fom_summary)
        print(red_summary)
        print(results['summary'])
        sys.stdout.flush()

    if args['--save']:
        # save U and URB for particular mu
        mu = {
            "D": [3e4, 2e4, 1e4],
            "E": [4 * fom.rhs_1.material.E_0],
            "NU": [0.3],
            "theta": 1.0,
        }
        U = fom.solve(mu)
        URB = reductor.reconstruct(rom.solve(mu))
        ERR = U - URB
        fom.visualize((U, URB, ERR), filename="results/meso.xdmf", legend=(
            "displacement", "u_rb", "error"))

if __name__ == "__main__":
    main(sys.argv[1:])
