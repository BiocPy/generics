import numpy as np
import pandas as pd
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


def test_combine_cols_mixed():
    num_rows = 20
    x = np.ones(shape=(num_rows, 10))
    y = sp.identity(num_rows)

    z = combine_cols(x, y)

    assert isinstance(z, np.ndarray)
    assert z.shape == (20, 30)


def test_pandas_dataframe():
    df1 = pd.DataFrame([["a", 1], ["b", 2]], columns=["letter", "number"])

    df2 = pd.DataFrame(
        [["c", 3, "cat"], ["d", 4, "dog"]], columns=["letter", "number", "animal"]
    )

    z = combine_cols(df1, df2)
    assert isinstance(z, pd.DataFrame)


def test_combine_cols_ndim():
    num_rows = 20
    x = np.ones(shape=(num_rows, 10, 20))
    y = np.ones(shape=(num_rows, 20, 20))

    z = combine_cols(x, y)

    assert isinstance(z, np.ndarray)
    assert z.shape == (20, 30, 20)
