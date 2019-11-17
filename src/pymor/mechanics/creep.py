#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dolfin as df
import numpy as np

from pymor.mechanics.history import HistoryMaterial
from pymor.parameters.base import Parametric


class CreepTestSolution:
    """analytical solution for a creep test with creep load sigma(t) = const.
    """

    def __init__(self, tau, sigma, mu):
        self.nkelem = mu['D'].size
        self.E0 = mu['E0']
        self.stiffness_moduli = [mu['D'][i] for i in range(self.nkelem)]
        self.tau = tau
        self.sig = sigma * mu['theta']

    def strain(self, time):
        spring = self.sig / self.E0 * np.ones(time.size)
        kelvin = np.zeros(time.size)
        for i in range(self.nkelem):
            kelvin += self.sig * (1.0 - np.exp(-time/self.tau[i])) / float(self.stiffness_moduli[i])
        return spring + kelvin

    def strain_limit(self):
        tmp = 0
        for i in range(self.nkelem):
            tmp += 1 / float(self.stiffness_moduli[i])
        return self.sig * (1/self.E0 + tmp)

    def stress_work(self):
        return self.strain_limit() * self.sig


class ParametricKelvinChain(Parametric, HistoryMaterial):
    """class representing parametric non-aging Kelvin Chain

    Parameters
    ----------
        mesh                    Partition of the domain Ω.
        degree                  Quadrature degree.
        parameters              A dict containing all material parameters.

    Attributes
    ----------
        dim                     Spatial dimension of the problem.
        nkelem                  Number of kelvin elements.
        stiffness_moduli        Elastic moduli of kelvin elements.
        tau                     Retardation times.
        E0                      Zero-th spring stiffness.
        dt                      Time step.
        plane_stress            Constraints to use in 2d case.
        parameter_setter        A function to set parameter value µ.
        parameter_type          Type of µ.
        lambda_1                First LAME constant (lambda).
        lambda_2                Second LAME constant (mu).
        scalar_space            Scalar quadrature space.
        tensor_space            Tensor valued quadrature space.
        history_variables       A dict holding all history variables.
        parameters              A dict containing all material parameters.
    """

    def __init__(self, mesh, degree, parameters=None):
        # TODO: ensure user specifies parameters correctly?
        assert isinstance(parameters['stiffness_moduli'], tuple)
        for m in parameters['stiffness_moduli']:
            assert isinstance(m, df.function.constant.Constant)
        for k in ('E0', 'dt'):
            assert isinstance(parameters[k], df.function.constant.Constant)

        self.nkelem = parameters.get('nkelem', 3)
        self.stiffness_moduli = parameters.get('stiffness_moduli')
        self.tau = parameters.get('tau', [3.0, 20.0, 40.0])
        self.E0 = parameters.get('E0')
        self.dt = parameters.get('dt')
        self.plane_stress = parameters.get('plane_stress', False)
        self.parameter_setter = parameters.get('parameter_setter', None)
        parameter_type = parameters.get('parameter_type', None)
        self.build_parameter_type(parameter_type)
        scalar_variables = []
        tensor_variables = ["sigma", "eps", "eps_prev", "csi"] + [f"gamma_{i}" for i in range(self.nkelem)]
        super().__init__(mesh, degree, scalar_variables, tensor_variables)
        self.initialize_history_variables()
        if self.dim == 1:
            NU = 0.0
        else:
            NU = 0.3
        self.lambda_1 = NU / ((1.0 + NU) * (1.0 - 2.0 * NU))
        self.lambda_2 = 1.0 / 2.0 / (1.0 + NU)
        self.parameters = parameters

        #  use like this:
        #  material = ParametricKelvinChain(mesh, degree, plane_stress=True)
        #  mat_r = type(material)(submesh, material.degree, parameters=material.parameters)

    def _set_mu(self, mu=None):
        mu = self.parse_parameter(mu)
        if self.parameter_setter:
            self.parameter_setter(mu)

    def get_beta(self):
        assert np.sum(self.tau) > 0, "You should set retardation times."
        assert float(self.dt) > 0, "You should set a time step dt."
        B = [df.exp(-self.dt / self.tau[i]) for i in range(self.nkelem)]
        return B

    def get_lambda(self):
        B = self.get_beta()
        L = [self.tau[i] * (1 - B[i]) / self.dt for i in range(self.nkelem)]
        return L

    def get_effective_stiffness(self, mu=None):
        if mu:
            self._set_mu(mu)
        lmbd = self.get_lambda()
        chain_compliance = 0
        for i in range(self.nkelem):
            chain_compliance += (1.0 - lmbd[i]) / self.stiffness_moduli[i]
        compliance = 1 / self.E0 + chain_compliance
        return 1 / compliance

    def Cx(self, E):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        return lambda_1 * df.tr(E) * df.Identity(3) + 2 * lambda_2 * E

    def elasticity(self, epsu, epsv):
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        return 2 * lambda_2 * df.inner(epsu, epsv) + lambda_1 * df.tr(epsu) * df.tr(epsv)

    def eps(self, disp):
        d = self.dim
        e = df.sym(df.grad(disp))
        if d == 1:
            return df.as_tensor([[e[0, 0], 0, 0],
                                 [0, 0, 0],
                                 [0, 0, 0]])
        elif d == 2 and not self.plane_stress:
            return df.as_tensor([[e[0, 0], e[0, 1], 0],
                                 [e[0, 1], e[1, 1], 0],
                                 [0, 0, 0]])
        elif d == 2 and self.plane_stress:
            ezz = -self.lambda_1 / (2.0 * self.lambda_2 + self.lambda_1) * (e[0, 0] + e[1, 1])
            return df.as_tensor([[e[0, 0], e[0, 1], 0],
                                 [e[0, 1], e[1, 1], 0],
                                 [0, 0, ezz]])
        elif d == 3:
            return e
        else:
            AttributeError("Spatial Dimension is not set.")

    def assemble_lhs(self, u, v, dx):
        """returns a dolfin.cpp.la.Matrix"""
        a = self.elasticity(self.eps(u), self.eps(v)) * dx
        return df.assemble(a)

    def get_nonaffine_form(self, v, dx):
        CE = self.Cx(self.history_variables["eps"] + self.history_variables["csi"])
        L = self.get_effective_stiffness() * df.inner(self.eps(v), CE) * dx - \
            df.inner(self.eps(v), self.history_variables["sigma"]) * dx
        return L

    def update_history(self, displacement, mu=None, range_space=None, source_dofs=np.s_[:]):
        """problem specific function to be provided by the user"""
        if isinstance(displacement, (df.cpp.la.PETScVector, np.ndarray)) and range_space is not None:
            V = range_space.V
            u = df.Function(V)
            uvec = u.vector()
            uvec[source_dofs] = displacement
            d = u
        elif isinstance(displacement, df.function.function.Function):
            # do nothing
            d = displacement
        else:
            raise KeyError
        #  self._set_mu(mu)

        # ### update strain
        self.history_variables["eps_prev"].assign(self.history_variables["eps"])
        strain = self.eps(d)
        self.local_project(strain, self.tensor_space, self.history_variables["eps"])

        # ### update stress
        stress = self.history_variables["sigma"] + float(self.get_effective_stiffness(mu)) * (
            self.Cx(self.history_variables["eps"]
                    - self.history_variables["eps_prev"]
                    - self.history_variables["csi"])
        )
        self.local_project(stress, self.tensor_space, self.history_variables["sigma"])

        # ### update gammas and csi
        csi = 0
        beta = self.get_beta()
        get_lambda = self.get_lambda()
        stiffness_moduli = self.stiffness_moduli
        for i in range(self.nkelem):
            new_gamma = (float(get_lambda[i]) * float(self.get_effective_stiffness(mu)) / stiffness_moduli[i] * (
                self.history_variables["eps"]
                - self.history_variables["eps_prev"]
                - self.history_variables["csi"]
            )
                + self.history_variables[f"gamma_{i}"] * float(beta[i]))
            self.local_project(new_gamma, self.tensor_space, self.history_variables[f"gamma_{i}"])
            csi += (1 - float(beta[i])) * self.history_variables[f"gamma_{i}"]
        self.history_variables["csi"].assign(csi)
