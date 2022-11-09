# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)
import numpy as np
import scipy.linalg as spla

from pymor.algorithms.projection import project
from pymor.algorithms.to_matrix import to_matrix
from pymor.core.cache import cached, CacheableObject
from pymor.models.iosys import LTIModel
from pymor.operators.numpy import NumpyHankelOperator, NumpyMatrixOperator


class ERAReductor(CacheableObject):
    cache_region = 'memory'

    def __init__(self, data, sampling_time):
        assert sampling_time > 0
        self.H = NumpyHankelOperator(data)
        self.__auto_init(locals())

    @cached
    def _s1_W1(self):
        self.logger.info('Computing output SVD ...')
        W1, s1, _ = spla.svd(np.hstack(self.H.markov_parameters), full_matrices=False)
        return s1, W1

    @cached
    def _s2_W2(self):
        self.logger.info('Computing input SVD ...')
        _, s2, W2 = spla.svd(np.vstack(self.H.markov_parameters), full_matrices=False)
        return s2, W2.conj().T

    @cached
    def _sv_U_V(self, l1, l2):
        H = self.H
        if l1:
            W1 = self.output_projector(l1)
            self.logger.info('Projecting Markov parameters ...')
            H = NumpyHankelOperator(W1.conj().T @ H.markov_parameters)
        if l2:
            W2 = self.input_projector(l2)
            self.logger.info('Projecting Markov parameters ...')
            H = NumpyHankelOperator(H.markov_parameters @ W2)

        self.logger.info(f'Computing SVD of the {"projected " if l1 or l2 else ""}Hankel matrix ...')
        U, sv, V = spla.svd(to_matrix(H), full_matrices=False)

        return sv, U.T, V

    def output_projector(self, l1):
        self.logger.info(f'Constructing output projector ({l1} tangential directions) ...')
        assert isinstance(l1, int) and l1 <= self.H.markov_parameters.shape[1]
        return self._s1_W1()[1][:, :l1]

    def input_projector(self, l2):
        self.logger.info(f'Constructing input projector ({l2} tangential directions) ...')
        assert isinstance(l2, int) and l2 <= self.H.markov_parameters.shape[2]
        return self._s2_W2()[1][:, :l2]

    def reduce(self, r=None, l1=None, l2=None, tol=None):
        assert r is not None or tol is not None
        _, p, m = self.H.markov_parameters.shape
        assert l1 is None or isinstance(l1, int) and l1 <= p
        assert l2 is None or isinstance(l2, int) and l2 <= m
        assert r is None or 0 < r <= min(self.H.range.dim * (l1 or p) / p, self.H.source.dim * (l2 or m) / m)

        sv, U, V = self._sv_U_V(l1, l2)
        sv, U, V = sv[:r], U[:r], V[:r]

        self.logger.info(f'Constructing reduced realization of order {r} ...')
        sqS = np.diag(np.sqrt(sv))
        Zo = U.T @ sqS
        A = NumpyMatrixOperator(spla.pinv(Zo[: -(l1 or p)]) @ Zo[(l1 or p):])
        B = NumpyMatrixOperator(sqS @ V[:, :(l2 or m)])
        C = NumpyMatrixOperator(Zo[:(l1 or p)])

        if l1:
            self.logger.info('Backprojecting tangential output directions ...')
            W1 = self.output_projector(l1)
            C = project(C, source_basis=None, range_basis=C.range.from_numpy(W1))
        if l2:
            self.logger.info('Backprojecting tangential input directions ...')
            W2 = self.input_projector(l2)
            B = project(B, source_basis=B.source.from_numpy(W2), range_basis=None)

        return LTIModel(A, B, C, sampling_time=self.sampling_time,
                        presets={'o_dense': np.diag(sv), 'c_dense': np.diag(sv)})
