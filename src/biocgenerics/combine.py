from functools import singledispatch
from typing import Any

from numpy import ndarray

from .combine_rows import combine_rows
from .combine_seqs import combine_seqs
from .utils import _is_1d_dense_arrays, _is_1d_sparse_arrays, _is_package_installed

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@singledispatch
def combine(*x: Any):
    """Combine vector-like objects (1-dimensional arrays).

    Custom classes may implement their own ``combine`` method.

    If the first element in ``x`` contains a ``combine`` method,
    the rest of the arguments are passed to that method.

    If all elements are 1-dimensional :py:class:`~numpy.ndarray`,
    we combine them using numpy's :py:func:`~numpy.concatenate`.

    If all elements are either 1-dimensional :py:class:`~scipy.sparse.spmatrix` or
    :py:class:`~scipy.sparse.sparray`, these objects are combined
    using scipy's :py:class:`~scipy.sparse.hstack`.

    If all elements are :py:class:`~pandas.Series` objects, they are combined using
    :py:func:`~pandas.concat`.

    For all other scenario's, all elements are coerced to a :py:class:`~list` and
    combined.

    Args:
        x (Any): Array of vector-like objects to combine.

            All elements of ``x`` are expected to be the same class or
            atleast compatible with each other.

    Raises:
        TypeError: If any object in the list cannot be coerced to a list.

    Returns:
        A combined object, typically the same type as the first element in ``x``.
        If the elements are a mix of dense and sparse objects, a :py:class:`~numpy.ndarray` is returned.
        A list if one of the objects is a list.
    """

    raise NotImplementedError("`combine` method is not implemented for objects.")


@combine.register(list)
def _combine_lists(*x: list):
    return combine_seqs(*x)


@combine.register(ndarray)
def _combine_dense_arrays(*x: ndarray):
    if _is_1d_dense_arrays(x) is True:
        return combine_seqs(*x)

    return combine_rows(*x)


if _is_package_installed("scipy") is True:
    import scipy.sparse as sp

    def _combine_sparse(*x):
        if _is_1d_sparse_arrays(x) is True:
            return combine_seqs(*x)

        return combine_rows(*x)

    combine_seqs.register(sp.sparray, _combine_sparse)
    combine_seqs.register(sp.spmatrix, _combine_sparse)


if _is_package_installed("pandas") is True:
    from pandas import DataFrame, Series

    @combine.register(Series)
    def _combine_series(*x):
        return combine_seqs(*x)

    @combine.register(DataFrame)
    def _combine_df(*x):
        return combine_seqs(*x)
