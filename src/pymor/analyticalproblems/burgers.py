# -*- coding: utf-8 -*-
# This file is part of the pyMor project (http://www.pymor.org).
# Copyright Holders: Felix Albrecht, Rene Milk, Stephan Rave
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from __future__ import absolute_import, division, print_function

import numpy as np

from pymor.core import Unpicklable, inject_sid
from pymor.tools import Named
from pymor.domaindescriptions import RectDomain, TorusDomain, BoundaryType
from pymor.functions import ConstantFunction, GenericFunction
from pymor.analyticalproblems.advection import InstationaryAdvectionProblem
from pymor.parameters.spaces import CubicParameterSpace


class BurgersProblem(InstationaryAdvectionProblem, Unpicklable):

    def __init__(self, vx = 1., vy = 1., torus=True, initial_data='sin', parameter_range=(1., 2.)):

        assert initial_data in ('sin', 'bump')

        def burgers_flux(U, mu):
            U = U.reshape(U.shape[:-1])
            U_exp = np.sign(U) * np.power(np.abs(U), mu['exponent'])
            R = np.empty(U.shape + (2,))
            R[...,0] = U_exp * vx
            R[...,1] = U_exp * vy
            return R
        inject_sid(burgers_flux, str(BurgersProblem) + '.burgers_flux')

        def burgers_flux_derivative(U, mu):
            U = U.reshape(U.shape[:-1])
            U_exp = mu['exponent'] * (np.sign(U) * np.power(np.abs(U), mu['exponent']-1))
            R = np.empty(U.shape + (2,))
            R[...,0] = U_exp * vx
            R[...,1] = U_exp * vy
            return R
        inject_sid(burgers_flux_derivative, str(BurgersProblem) + '.burgers_flux_derivative')

        flux_function = GenericFunction(burgers_flux, dim_domain=1, shape_range=(2,),
                                        parameter_type={'exponent': 0},
                                        name='burgers_flux')

        flux_function_derivative = GenericFunction(burgers_flux_derivative, dim_domain=1, shape_range=(2,),
                                                   parameter_type={'exponent': 0},
                                                   name='burgers_flux')

        if initial_data == 'sin':
            def initial_data(x):
                return 0.5 * (np.sin(2 * np.pi * x[..., 0]) * np.sin(2 * np.pi * x[..., 1]) + 1.)
            inject_sid(initial_data, str(BurgersProblem) + '.initial_data_sin')
            dirichlet_data=ConstantFunction(dim_domain=2, value=0.5)
        else:
            def initial_data(x):
                return (x[..., 0] >= 0.5) * (x[..., 0] <= 1) * 1
            inject_sid(initial_data, str(BurgersProblem) + '.initial_data_riemann')
            dirichlet_data=ConstantFunction(dim_domain=2, value=0)

        initial_data = GenericFunction(initial_data, dim_domain=2)

        domain = TorusDomain([[0, 0], [2, 1]]) if torus else RectDomain([[0, 0], [2, 1]], right=None,
                top=None)

        super(BurgersProblem, self).__init__(domain=domain,
                                             rhs=None,
                                             flux_function=flux_function,
                                             flux_function_derivative=flux_function_derivative,
                                             initial_data=initial_data,
                                             dirichlet_data=dirichlet_data,
                                             T=0.3, name='Burgers Problem')

        self.parameter_space = CubicParameterSpace({'exponent': 0}, *parameter_range)
