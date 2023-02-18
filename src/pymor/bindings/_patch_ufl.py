# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

from pymor.core.defaults import defaults


@defaults('doit')
def patch_ufl(doit=True):
    """Monkey patch ufl.algorithms.estimate_total_polynomial_degree.

    Catches `TypeError`, which can be called by certain UFL expressions, and returns
    `default_degree`.

    This is needed, for instance, when using :mod:`pymor.discretizers.fenics` on a
    :func:`~pymor.analyticalproblems.thermalblock.thermal_block_problem`.
    """
    if not doit:
        return

    import ufl

    real_estimate_total_polynomial_degree = ufl.algorithms.estimate_total_polynomial_degree

    def estimate_total_polynomial_degree_wrapper(e, default_degree=1, element_replace_map={}):
        try:
            return real_estimate_total_polynomial_degree(e, default_degree=default_degree,
                                                         element_replace_map=element_replace_map)
        except TypeError:
            return default_degree

    ufl.algorithms.estimate_degrees.estimate_total_polynomial_degree = estimate_total_polynomial_degree_wrapper
    ufl.algorithms.estimate_total_polynomial_degree = estimate_total_polynomial_degree_wrapper

    # use sys.modules for monkey patching since compute_form_data is at the same time function
    # and sub-module
    import sys
    sys.modules['ufl.algorithms.compute_form_data'].estimate_total_polynomial_degree \
        = estimate_total_polynomial_degree_wrapper
