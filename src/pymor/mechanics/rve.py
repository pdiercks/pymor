# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config

if config.HAVE_FENICS:
    import ufl
    import dolfin as df
    from pymor.operators.constructions import LincombOperator, VectorOperator
    from pymor.parameters.base import ParameterType
    from pymor.parameters.functionals import (
        ProjectionParameterFunctional,
        ExpressionParameterFunctional,
        ProductParameterFunctional
    )

    def get_linear_elastic_macro_strain_operator(range_space, measure):
        r"""Returns a |LincombOperator| which represents the parameter separated form of

        .. math::
            \int_{\Omega} \varepsilon(v) \cdot C \cdot E \; \mathrm{d}x,

        where the 6 independent components of the macroscopic strain :math:`E`
        and young's modulus and poisson ratio of the material are used as parameters.
        """

        family = range_space.V.ufl_element(),family()
        assert family in ('Lagrange', 'Discontinuous Lagrange', 'Mixed')
        assert isinstance(measure, ufl.measure.Measure)

        gdim = range_space.V.mesh().geometric_dimension()
        assert gdim in (2, 3)
        ncomponents = 3
        if gdim == 3:
            ncomponents *= 2

        pt = ParameterType({'EPS': (ncomponents,), 'E': (), 'NU': ()})

        # ### Lame constants as parameter functionals
        L = ExpressionParameterFunctional("E * NU / ((1.0 + NU) * (1.0 - 2.0 * NU))", pt) # lambda
        M = ExpressionParameterFunctional("E / 2.0 / (1.0 + NU)", pt)                     # mu

        # ### trace of E as parameter functional
        expression = ' + '.join([f'EPS[{i}]' for i in range(gdim)])
        trE = ExpressionParameterFunctional(expression, pt)

        # ### parameter functional for 1st term
        LtrE = ProductParameterFunctional([L, trE])

        # ### µ_ij as parameter functionals
        mu_ij = [ProjectionParameterFunctional(component_name='EPS',
                                               component_shape=(ncomponents,),
                                               coordinates=(c,))
                 for c in range(ncomponents)]

        # ### list of (product) parameter functionals for 6 forms resulting from 2nd term
        M_mu_ij = [ProductParameterFunctional([M, mu]) for mu in mu_ij]

        # ### gather all parameter functionals
        thetas = [LtrE] + M_mu_ij

        # ### assemble vectors
        if family == 'Mixed':
            v, lambda_v = df.TestFunctions(range_space.V)
        else:
            v = df.TestFunction(range_space.V)
        W = df.sym(df.grad(v))
        vector = []

        # ### caution! the following (voigt) mapping is used
        #  in the 2d case:
        #  E_ij = [[EPS[0], EPS[2], 0.0],
        #          [EPS[2], EPS[1], 0.0],
        #          [0.0, 0.0, 0.0]]
        #  in the 3d case:
        #  E_ij = [[EPS[0], EPS[3], EPS[4]],
        #          [EPS[3], EPS[1], EPS[5]],
        #          [EPS[4], EPS[5], EPS[2]]]

        for i in range(gdim):
            vector.append(df.assemble(2.0 * W[i, i] * measure))

        for i in range(gdim):
            for j in range(gdim):
                if i < j:
                    vector.append(df.assemble(4.0 * W[i, j] * measure))

        vector_operators = [VectorOperator(FenicsVectorSpace(range_space.V).make_array([v])) for v
                            in vector]

        return LincombOperator(vector_operators, thetas)
