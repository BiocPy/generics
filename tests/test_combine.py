import numpy as np
from biocgenerics.combine import combine
from scipy import sparse as sp

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_combine_basic_list():
    x = [1, 2, "c"]
    y = ["a", "b"]

    z = combine(x, y)

    assert z == x + y
    assert isinstance(z, list)
    assert len(z) == len(x) + len(y)


def test_combine_basic_dense():
    x = [1, 2, 3]
    y = [0.1, 0.2]
    xd = np.array([1, 2, 3])
    yd = np.array([0.1, 0.2], dtype=float)

    zcomb = combine(xd, yd)

    z = x + y
    zd = np.array(z)

    assert all(np.isclose(zcomb, zd)) is True
    assert isinstance(zcomb, np.ndarray)
    assert len(zcomb) == len(zd)


def test_combine_basic_mixed_dense_list():
    x = [1, 2, 3]
    y = [0.1, 0.2]
    xd = np.array([1, 2, 3])

    zcomb = combine(xd, y)

    z = x + y

    assert zcomb == z
    assert isinstance(zcomb, list)
    assert len(zcomb) == len(xd) + len(y)


def test_combine_basic_mixed_tuple_list():
    x = [1, 2, 3]
    y = (0.1, 0.2)
    xd = np.array([1, 2, 3])

    zcomb = combine(xd, y, x)

    z = x + list(y) + x

    assert zcomb == z
    assert isinstance(zcomb, list)
    assert len(zcomb) == 2 * len(xd) + len(y)


def test_combine_basic_sparse():
    x = np.array([1, 2, 3])
    y = np.array([0.1, 0.2])

    sx = sp.csr_array(x)
    sy = sp.csr_array(y)

    z = combine(sx, sy)

    assert isinstance(z, sp.spmatrix)
    assert z.shape[1] == len(x) + len(y)

    # mixed sparse arrays
    sx = sp.csr_array(x)
    sy = sp.coo_array(y)

    z = combine(sx, sy)

    assert isinstance(z, sp.spmatrix)
    assert z.shape[1] == len(x) + len(y)
