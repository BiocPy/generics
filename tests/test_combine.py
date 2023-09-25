from biocgenerics.combine import combine
import numpy as np
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


def test_combine_basic_numpy():
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

def test_combine_basic_sparse():
    x = [1, 2, 3]
    y = [0.1, 0.2]
    xd = np.array([1, 2, 3])

    zcomb = combine(xd, y)

    z = x + y

    assert zcomb == z
    assert isinstance(zcomb, list)
    assert len(zd) == len(xd) + len(yd)