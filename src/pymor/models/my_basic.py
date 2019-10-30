# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2019 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

from pymor.models.interfaces import ModelInterface
from pymor.operators.constructions import VectorOperator, induced_norm
from pymor.operators.interfaces import OperatorInterface
from pymor.tools.frozendict import FrozenDict
from pymor.vectorarrays.interfaces import VectorArrayInterface


class ModelBase(ModelInterface):
    """Base class for |Models| providing some common functionality."""

    sid_ignore = ModelInterface.sid_ignore | {'visualizer'}

    def __init__(self, products=None, estimator=None, visualizer=None,
                 cache_region=None, name=None, **kwargs):

        self.products = FrozenDict(products or {})
        self.estimator = estimator
        self.visualizer = visualizer
        self.enable_caching(cache_region)
        self.name = name

        if products:
            for k, v in products.items():
                setattr(self, f'{k}_product', v)
                setattr(self, f'{k}_norm', induced_norm(v))

    def visualize(self, U, **kwargs):
        """Visualize a solution |VectorArray| U.

        Parameters
        ----------
        U
            The |VectorArray| from
            :attr:`~pymor.models.interfaces.ModelInterface.solution_space`
            that shall be visualized.
        kwargs
            See docstring of `self.visualizer.visualize`.
        """
        if self.visualizer is not None:
            self.visualizer.visualize(U, self, **kwargs)
        else:
            raise NotImplementedError('Model has no visualizer.')

    def estimate(self, U, mu=None):
        if self.estimator is not None:
            return self.estimator.estimate(U, mu=mu, m=self)
        else:
            raise NotImplementedError('Model has no estimator.')


class InstationaryModel(ModelBase):
    """Generic class for models of instationary problems.

    This class describes instationary problems given by the equations::

                             L(μ) = F(u(µ), t, μ)
                          u(0, μ) = u_0(μ)

    for t in [0,T], where L is a linear affine |Operator|, F consists
    of a non-affine |Operator| depending on the previous solution and a
    time-dependent linear affine vector-like |Operator|, and u_0 the
    initial data.

    Parameters
    ----------
    T
        The final time T.
    initial_data
        The initial data `u_0`. Either a |VectorArray| of length 1 or
        (for the |Parameter|-dependent case) a vector-like |Operator|
        (i.e. a linear |Operator| with `source.dim == 1`) which
        applied to `NumpyVectorArray(np.array([1]))` will yield the
        initial data for a given |Parameter|.
    operator
        The |Operator| L.
    rhs_0
        The linear and affine part of the right-hand side F.
    rhs_1
        The non-affine part of the right-hand side F.
    num_values
        The number of returned vectors of the solution trajectory. If `None`, each
        intermediate vector that is calculated is returned.
    outputs
        A dict of additional output |Functionals| associated with the model.
    products
        A dict of product |Operators| defined on the discrete space the
        problem is posed on. For each product a corresponding norm
        is added as a method of the model.
    parameter_space
        The |ParameterSpace| for which the discrete problem is posed.
    estimator
        An error estimator for the problem. This can be any object with
        an `estimate(U, mu, m)` method. If `estimator` is
        not `None`, an `estimate(U, mu)` method is added to the
        model which will call `estimator.estimate(U, mu, self)`.
    visualizer
        A visualizer for the problem. This can be any object with
        a `visualize(U, m, ...)` method. If `visualizer`
        is not `None`, a `visualize(U, *args, **kwargs)` method is added
        to the model which forwards its arguments to the
        visualizer's `visualize` method.
    cache_region
        `None` or name of the |CacheRegion| to use.
    name
        Name of the model.

    Attributes
    ----------
    T
        The final time T.
    initial_data
        The intial data u_0 given by a vector-like |Operator|. The same
        as `operators['initial_data']`.
    operator
        The |Operator| L. The same as `operators['operator']`.
    rhs_0
        The linear and affine part of the right-hand side F. The same as `operators['rhs_0']`.
    rhs_1
        The non-affine part of the right-hand side F. The same as `operators['rhs_1']`.
    outputs
        Dict of all output |Functionals|.
    products
        Dict of all product |Operators| associated with the model.
    """

    def __init__(self, T, initial_data, operator, rhs_0, rhs_1, mass=None, num_values=None,
                 outputs=None, products=None, parameter_space=None, estimator=None, visualizer=None,
                 cache_region=None, name=None):

        if isinstance(rhs_0, VectorArrayInterface):
            assert rhs_0 in operator.range
            rhs_0 = VectorOperator(rhs_0, name='rhs_0')

        if isinstance(initial_data, VectorArrayInterface):
            assert initial_data in operator.source
            initial_data = VectorOperator(initial_data, name='initial_data')

        assert initial_data.source.is_scalar

        assert rhs_0 is None \
            or rhs_0.linear and rhs_0.range == operator.range and rhs_0.source.is_scalar
        #  assert rhs_1 is None \
        #      or rhs_1.linear and rhs_1.range == operator.range and rhs_1.source == operator.source
        # rhs_1.linear = False in case of FenicsOperator
        assert rhs_1 is None \
            or rhs_1.range == operator.range and rhs_1.source == operator.source

        super().__init__(products=products, estimator=estimator,
                         visualizer=visualizer, cache_region=cache_region, name=name)
        self.T = T
        self.initial_data = initial_data
        self.operator = operator
        self.rhs_0 = rhs_0
        self.rhs_1 = rhs_1
        self.mass = mass
        self.solution_space = self.operator.source
        self.num_values = num_values
        self.outputs = FrozenDict(outputs or {})
        self.build_parameter_type(self.initial_data, self.operator, self.rhs_0, self.rhs_1, provides={'_t': 0})
        self.parameter_space = parameter_space

    def with_time_stepper(self, **kwargs):
        raise NotImplementedError
        return self.with_(time_stepper=self.time_stepper.with_(**kwargs))

    def _solve(self, mu=None, **kwargs):
        mu = self.parse_parameter(mu).copy()

        onset_of_unloading = kwargs.get('onset_of_unloading', 1)
        return_rhs = kwargs.get('return_rhs', False)
        return_stress = kwargs.get('return_stress', False)
        return_strain = kwargs.get('return_strain', False)

        # explicitly checking if logging is disabled saves the expensive str(mu) call
        if not self.logging_disabled:
            self.logger.info(f'Solving {self.name} for {mu} ...')

        A = self.operator
        F0 = self.rhs_0
        F1 = self.rhs_1
        if hasattr(F1, 'restricted_operator'):
            # TODO: if this is True then get the restricted material for submesh method
            try:
                material = F1.restricted_operator.operator.material  # assemble_local
            except AttributeError:
                material = F1.restricted_operator.op.material  # submesh
        else:
            material = F1.material
        assert isinstance(self.operator, OperatorInterface)  # move this to init?
        assert isinstance(self.rhs_0, (type(None), OperatorInterface, VectorArrayInterface))
        assert isinstance(self.rhs_1, (type(None), OperatorInterface))
        assert self.operator.source == self.operator.range
        #  dt = material._dt
        dt = mu['dt']
        nt = int(self.T / dt)
        #  num_values = self.num_values or nt + 1
        mu['_t'] = 0

        material.initialize_history_variables()
        U0 = self.initial_data.as_range_array(mu)  # instantaneous elasticity

        R = A.source.empty(reserve=nt+1)
        R.append(U0)

        t = 0
        loading = 1
        U = U0.copy()

        data = {}

        if return_rhs:
            data['rhs'] = self.operator.range.empty()

        if return_stress:
            data['stress'] = list()

        if return_strain:
            total_strain = material.history_variables["eps"].copy(deepcopy=True)
            data['strain'] = list()

        for n in range(nt):  # TODO: adaptive time stepping
            t += dt
            mu['_t'] = t

            rhs = F1.apply(U, mu=mu)  # history update is done in FenicsOperator.apply

            if return_rhs:
                data['rhs'].append(rhs)  # return only non-affine part

            if return_stress:
                sig = material.history_variables["sigma"].copy(deepcopy=True)
                data['stress'].append(sig)

            if return_strain:
                eps = material.history_variables["eps"]
                csi = material.history_variables["csi"]
                material.local_project(eps+csi, material.tensor_space, total_strain)
                data['strain'].append(total_strain.copy(deepcopy=True))

            if t > self.T / onset_of_unloading:  # TODO: make onset of unloading an argument to solve?
                loading = 0  # unloading

            rhs += F0.as_range_array(mu=mu) * loading

            # ### solve
            U = A.apply_inverse(rhs, mu=mu)
            R.append(U)

        if return_rhs:
            return R, data
        else:
            return R
