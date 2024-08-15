import dolfinx as df
import numpy as np
import ufl
from mpi4py import MPI

from pymor.bindings.fenicsx import BCGeom, FenicsxMatrixBasedOperator
from pymor.tools.random import get_rng
from pymor.vectorarrays.numpy import NumpyVectorSpace


def test(nx, ny, degree, value_shape):
    domain = df.mesh.create_unit_square(MPI.COMM_WORLD, nx, ny)
    gdim = domain.geometry.dim
    V = df.fem.functionspace(domain, ('P', degree, value_shape))
    ndofs = V.dofmap.bs * V.dofmap.index_map.size_local
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    params = {'R': 1}
    coeff = df.fem.Function(V, name='coeff')
    match value_shape:
        case ():
            form = ufl.inner(u, v) * coeff * ufl.dx

            def param_setter(mu):
                value = mu['R']
                coeff.interpolate(lambda x: x[0] * value)

        case (gdim,):
            grad_u = ufl.grad(u)
            grad_v = ufl.grad(v)
            F = ufl.grad(coeff) + ufl.Identity(gdim)
            i, j, k = ufl.indices(3)
            form = grad_v[i, j] * grad_u[j, k] * F[k, i] * ufl.dx

            def param_setter(mu):
                value = mu['R']
                coeff.interpolate(lambda x: (x[0] * value, x[1] * value))

        case _:
            raise NotImplementedError

    def bottom(x):
        return np.isclose(x[1], 0.0)

    if len(value_shape):
        dirichlet_value = (df.default_scalar_type(5.4),) * value_shape[0]
    else:
        dirichlet_value = df.default_scalar_type(5.4)
    u_D = df.fem.Constant(domain, dirichlet_value)
    bc_geom = BCGeom(u_D, bottom, V)

    operator = FenicsxMatrixBasedOperator(form, params, param_setter=param_setter, bcs=(bc_geom,))

    # pick magic dofs at random
    magic_dofs = set()
    nmagic = 13
    rng = get_rng()
    while len(magic_dofs) < nmagic:
        magic_dofs.add(rng.integers(0, ndofs))
    magic_dofs = np.array(sorted(magic_dofs), dtype=np.int32)
    r_op, r_source_dofs = operator.restricted(magic_dofs)

    red_coeffs = r_op.op.form.coefficients()
    assert len(red_coeffs) == 1
    assert red_coeffs[0].name == 'coeff'

    nt = 10 # number of test vectors
    def compare(mu) -> bool:
        U = operator.source.random(nt)
        AU = operator.apply(U, mu)

        U_dofs = NumpyVectorSpace.make_array(U.dofs(r_source_dofs))
        r_AU = r_op.apply(U_dofs, mu=mu)

        is_zero = np.sum(np.abs(AU.to_numpy()[:, magic_dofs] - r_AU.to_numpy()))
        return is_zero < 1e-9

    ntest = 10 # size of parameter test set
    mus = operator.parameters.space({'R': (0.1, 10.0)}).sample_randomly(ntest)
    success = []
    for mu in mus:
        success.append(compare(mu))

    if all(success):
        print(f'test passed for {degree=}, {value_shape=}')


if __name__ == '__main__':
    test(10, 10, 1, ())
    test(10, 10, 2, ())
    test(10, 10, 1, (2,))
    test(10, 10, 2, (2,))
