#!/usr/bin/env python
# -*- coding: utf-8 -*-

import dolfin as df


class HistoryMaterial:
    """class to equip material with quadrature point data

    Parameters
    ----------
        mesh                    Partition of the domain Ω.
        degree                  Quadrature degree.
        scalar_variables        Scalar history variables given as list of str.
        tensor_variables        Tensorial history variables given as list of str.

    Attributes
    ----------
        dim                     Spatial dimension of the problem.
        degree                  Quadrature degree.
        scalar_space            Scalar quadrature space.
        tensor_space            Tensor valued quadrature space.
        history_variables       Dict holding history variables.
    """

    def __init__(self, mesh, degree, scalar_variables, tensor_variables):
        self.dim = mesh.topology().dim()
        self.degree = degree
        self.scalar_variables = scalar_variables
        self.tensor_variables = tensor_variables
        FE = df.FiniteElement("Quadrature", mesh.ufl_cell(), degree=degree,
                              quad_scheme="default")
        self.scalar_space = df.FunctionSpace(mesh, FE)
        TE = df.TensorElement("Quadrature", mesh.ufl_cell(), degree=degree,
                              quad_scheme="default", symmetry=True, shape=(3, 3))
        self.tensor_space = df.FunctionSpace(mesh, TE)

    def local_project(self, v, V, u=None):
        """project v onto V and store the values in u"""
        if self.degree > 1:  # use this because of bug in uflacs representation ...
            df.parameters["form_compiler"]["representation"] = "quadrature"
        metadata = {"quadrature_degree": self.degree, "quadrature_scheme": "default"}
        dx = df.dx(metadata=metadata)
        v_trial = df.TrialFunction(V)
        v_test = df.TestFunction(V)
        A = df.inner(v_trial, v_test) * dx
        b = df.inner(v, v_test) * dx
        solver = df.LocalSolver(A, b)
        solver.factorize()
        if u is None:
            u = df.Function(V)
            solver.solve_local_rhs(u)
            if self.degree > 1:
                df.parameters["form_compiler"]["representation"] = "uflacs"
            return u
        else:
            solver.solve_local_rhs(u)
            if self.degree > 1:
                df.parameters["form_compiler"]["representation"] = "uflacs"
            return

    def initialize_history_variables(self):
        scalar_variables = self.scalar_variables
        tensor_variables = self.tensor_variables
        assert hasattr(self, "scalar_space") and hasattr(
            self, "tensor_space"), "Quadrature Spaces are not set up correctly."
        assert isinstance(scalar_variables, list) and all(isinstance(s, str) for s in
                                                          scalar_variables)
        assert isinstance(tensor_variables, list) and all(isinstance(t, str) for t in
                                                          tensor_variables)
        if not hasattr(self, "history_variables"):
            values = [df.Function(self.scalar_space, name=s) for s in scalar_variables] +\
                     [df.Function(self.tensor_space, name=t) for t in tensor_variables]
            keys = scalar_variables + tensor_variables
            self.history_variables = {k: v for (k, v) in zip(keys, values)}
        NULL = df.Constant((
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0)))
        for s in scalar_variables:
            self.history_variables[s].assign(df.Constant(0.0))
        for t in tensor_variables:
            self.history_variables[t].assign(NULL)
