import numpy as np
from biocgenerics.combine_cols import combine_cols
from scipy import sparse as sp

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def test_combine_cols_dense():
    num_rows = 20
    x = np.ones(shape=(num_rows, 10))
    y = np.random.rand(num_rows, 5)

    z = combine_cols(x, y)

    assert isinstance(z, np.ndarray)
    assert z.shape == (20, 15)


def test_combine_cols_sparse():
    num_rows = 20

    x = sp.random(num_rows, 10)
    y = sp.identity(num_rows)

    z = combine_cols(x, y)

    assert isinstance(z, sp.spmatrix)
    assert z.shape == (20, 30)


# def test_combine_cols_mixed():
#     num_cols = 20
#     x = np.ones(shape=(10, num_cols))
#     y = sp.identity(num_cols)

#     z = combine(x, y)

#     assert isinstance(z, sp.spmatrix)
#     assert z.shape == (30, 20)
