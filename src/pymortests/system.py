# This file is part of the pyMOR project (http://www.pymor.org).
# Copyright 2013-2020 pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (http://opensource.org/licenses/BSD-2-Clause)

import numpy as np

from pymor.algorithms.system import project_system
from pymor.operators.constructions import ZeroOperator, VectorOperator
from pymor.operators.numpy import NumpyMatrixOperator
from pymor.operators.block import BlockOperator, UnblockableBlockOperator
from pymor.vectorarrays.numpy import NumpyVectorSpace
from pymor.algorithms.gram_schmidt import gram_schmidt


def test_project_system():
    # block operator
    A, B, C, D = (
        NumpyMatrixOperator(np.eye(5)), NumpyMatrixOperator(
            np.eye(5) * 2), None, NumpyMatrixOperator(np.eye(5) * 3)
    )
    op = BlockOperator([[A, B], [C, D]])

    # basis
    U, V = gram_schmidt(NumpyVectorSpace(5).random(1, seed=123)), gram_schmidt(
        NumpyVectorSpace(5).random(1, seed=456))

    # block rhs
    rhs_blocks = (U, V)
    rhs = VectorOperator(op.source.make_array(rhs_blocks))

    pop = project_system(op, [U, V], [U, V])
    prhs = project_system(rhs, [U, V], None)
    a, b, c, d = pop.blocks.ravel()

    assert np.max(np.abs(a.matrix - np.eye(len(U)))) < 1e-15
    assert np.max(np.abs(b.matrix - U.inner(V) * 2)) < 1e-15
    assert isinstance(c, ZeroOperator)
    assert np.max(np.abs(d.matrix - np.eye(len(V))*3)) < 1e-15

    # FOM
    u = op.apply_inverse(rhs.array)
    assert np.allclose(u.to_numpy(), np.hstack(
        (U.to_numpy() - 2 / 3 * V.to_numpy(), 1 / 3 * V.to_numpy())))

    # ROM
    v = pop.apply_inverse(prhs.array)
    assert np.allclose(v.to_numpy(), np.hstack(
        (1 - 2 / 3 * U.inner(V), np.array([[1 / 3]]))))

    # projection wrt only one block
    pop = project_system(op, [None, V], [None, V])
    prhs = project_system(rhs, [None, V], None)
    a, b, c, d = pop.blocks.ravel()

    assert np.max(np.abs(a.matrix - A.matrix)) < 1e-15
    assert np.max(np.abs(b.matrix - V.to_numpy().T * 2)) < 1e-15
    assert isinstance(c, ZeroOperator)
    assert np.max(np.abs(d.matrix - np.eye(len(V))*3)) < 1e-15

    e, f = prhs.array._blocks

    assert np.allclose(e.to_numpy(), U.to_numpy())
    assert np.allclose(f.to_numpy(), V.inner(V))


def test_UnblockableBlockOperator():
    # block operator
    a, b, c, d = (
        np.eye(5), 2 * np.eye(5), np.zeros((5, 5)), 3 * np.eye(5)
    )
    A, B, C, D = (
        NumpyMatrixOperator(np.eye(5)), NumpyMatrixOperator(
            np.eye(5) * 2), None, NumpyMatrixOperator(np.eye(5) * 3)
    )
    op = UnblockableBlockOperator([[A, B], [C, D]])

    nop = op._unblock()
    assert np.allclose(nop.matrix, np.block([[a, b], [c, d]]))

    V = NumpyVectorSpace(10).ones()
    U = op.apply_inverse(V)

    assert np.allclose(U.to_numpy().flatten(), np.ones(10) / 3)
