# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.core.config import config


if config.HAVE_FENICS:
    import inspect
    import dolfin as df
    import ufl
    import numpy as np
    from fenics_modules.helpers import local_project

    from pymor.bindings.fenics import FenicsVector, FenicsVectorSpace, FenicsMatrixOperator
    from pymor.core.defaults import defaults
    from pymor.core.interfaces import BasicInterface
    from pymor.operators.basic import OperatorBase
    from pymor.operators.numpy import NumpyMatrixOperator
    from pymor.vectorarrays.interfaces import _create_random_values
    from pymor.vectorarrays.list import CopyOnWriteVector, ListVectorSpace
    from pymor.vectorarrays.numpy import NumpyVectorSpace

    class LinearStrainOperator(OperatorBase):
        """Computes the linear strain tensor at quadrature points from
        a given displacement solution"""

        linear = True

        def __init__(self, source_space, range_space, name=None):
            self.source = source_space
            self.range = range_space
            self.gdim = source_space.V.mesh().geometric_dimension()
            self.name = name

        def apply(self, U, mu=None):
            assert U in self.source
            tmp = df.Function(self.source.V)
            R = []
            for u in U._list:
                tmp[:] = u.impl
                e = self.eps(tmp)
                E = local_project(e, self.range.V)
                R.append(E)
            return self.range.make_array(R)

        def eps(self, disp):
            """auxiliary function to compute eps"""
            d = self.gdim
            e = df.sym(df.grad(disp))
            if d == 1:
                return df.as_tensor([[e[0, 0], 0, 0],
                                     [0, 0, 0],
                                     [0, 0, 0]])
            elif d == 2 and not self.plane_stress:
                return df.as_tensor([[e[0, 0], e[0 ,1], 0],
                                     [e[0, 1], e[1, 1], 0],
                                     [0, 0, 0]])
            elif d == 2 and self.plane_stress:
                ezz = -self.lambda_1 / (2.0 * self.lambda_2 + self.lambda_1)\
                        * (e[0, 0] + e[1, 1])
                return df.as_tensor([[e[0, 0], e[0 ,1], 0],
                                     [e[0, 1], e[1, 1], 0],
                                     [0, 0, ezz]])
            elif d == 3:
                return e
            else:
                AttributeError("Spatial Dimension is not set.")
