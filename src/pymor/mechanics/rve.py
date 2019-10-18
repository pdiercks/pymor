# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

if config.HAVE_FENICS:
    import ufl
    import dolfin as df
    import numpy as np
    from pymor.operators.constructions import LincombOperator, VectorOperator
    from pymor.parameters.base import ParameterType
    from pymor.parameters.functionals import (
        ProjectionParameterFunctional,
        ExpressionParameterFunctional,
        ProductParameterFunctional
    )

    def get_linear_elastic_macro_strain_operator(range_space, parameter_type, measure, subdomain_ids):
        r"""Returns a |LincombOperator| which represents the parameter separated form of

        .. math::
            \int_{\Omega} \varepsilon(v) \cdot C \cdot E \; \mathrm{d}x,

        where the 6 independent components of the macroscopic strain :math:`E`
        and young's modulus and poisson ratio of the material are used as parameters.
        """
        assert isinstance(subdomain_ids, tuple)
        assert subdomain_ids[0] > 0, "marking should start from 1, which is usually the case for gmsh-generated meshes"

        NOMEGA = len(subdomain_ids)
        shapes = {'EPS': ((3,), (6,)), 'E': ((NOMEGA,),), 'NU': ((NOMEGA,),)}

        family = range_space.V.ufl_element().family()
        assert family in ('Lagrange', 'Discontinuous Lagrange', 'Mixed')
        if family == 'Mixed':
            v_test, lambda_v = df.TestFunctions(range_space.V)
        else:
            v_test = df.TestFunction(range_space.V)
        gdim = range_space.V.mesh().geometric_dimension()
        assert gdim in (2, 3)

        pt = parameter_type
        for k in pt.keys():
            assert k in ('EPS', 'E', 'NU')
            assert pt[k] in shapes[k]
        assert isinstance(measure, ufl.measure.Measure)

        # ### caution! the following (voigt) mapping is used
        #  in the 2d case:
        #  E_ij = [[EPS[0], EPS[2], 0.0],
        #          [EPS[2], EPS[1], 0.0],
        #          [0.0, 0.0, 0.0]]
        #  in the 3d case:
        #  E_ij = [[EPS[0], EPS[3], EPS[4]],
        #          [EPS[3], EPS[1], EPS[5]],
        #          [EPS[4], EPS[5], EPS[2]]]

        vector = []
        thetas = []

        for ID in subdomain_ids:
            # ### Lame constants as parameter functionals
            L = ExpressionParameterFunctional(
                f"E[{int(ID-1)}] * NU[{int(ID-1)}] / ((1.0 + NU[{int(ID-1)}]) * (1.0 - 2.0 * NU[{int(ID-1)}]))", pt) # lambda
            M = ExpressionParameterFunctional(f"E[{int(ID-1)}] / 2.0 / (1.0 + NU[{int(ID-1)}])", pt)       # mu

            # ### trace of E as parameter functional
            expression = ' + '.join([f'EPS[{i}]' for i in range(gdim)])
            trE = ExpressionParameterFunctional(expression, pt)

            # ### parameter functional for 1st term
            LtrE = ProductParameterFunctional([L, trE])

            # ### µ_ij as parameter functionals
            mu_ij = [ProjectionParameterFunctional(component_name='EPS',
                                                   component_shape=(pt['EPS'][0],),
                                                   coordinates=(c,))
                     for c in range(pt['EPS'][0])]

            # ### list of (product) parameter functionals for 6 forms resulting from 2nd term
            M_mu_ij = [ProductParameterFunctional([M, mu]) for mu in mu_ij]

            # ### gather all parameter functionals
            thetas += [LtrE] + M_mu_ij

            # ### assemble vectors
            W = df.sym(df.grad(v_test))
            vector.append(df.assemble(df.tr(W) * measure(ID)))

            for i in range(gdim):
                vector.append(df.assemble(2.0 * W[i, i] * measure(ID)))

            for i in range(gdim):
                for j in range(gdim):
                    if i < j:
                        vector.append(df.assemble(4.0 * W[i, j] * measure(ID)))

        # turn all vectors into VectorOperators
        vector_operators = [VectorOperator(range_space.make_array([v])) for v in vector]

        return LincombOperator(vector_operators, thetas)
