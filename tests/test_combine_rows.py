import numpy as np
from biocgenerics.combine_rows import combine_rows
from scipy import sparse as sp

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_combine_rows_dense():
    num_cols = 20
    x = np.ones(shape=(10, num_cols))
    y = np.random.rand(5, num_cols)

    z = combine_rows(x, y)

    assert isinstance(z, np.ndarray)
    assert z.shape == (15, 20)


def test_combine_rows_sparse():
    num_cols = 20

    x = sp.random(10, num_cols)
    y = sp.identity(num_cols)

    z = combine_rows(x, y)

    assert isinstance(z, sp.spmatrix)
    assert z.shape == (30, 20)


def test_combine_rows_mixed():
    num_cols = 20
    x = np.ones(shape=(10, num_cols))
    y = sp.identity(num_cols)

    z = combine_rows(x, y)

    assert isinstance(z, sp.spmatrix)
    assert z.shape == (30, 20)