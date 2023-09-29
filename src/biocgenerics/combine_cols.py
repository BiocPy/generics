from functools import singledispatch
from typing import Any

from numpy import ndarray
from scipy.sparse import spmatrix

from .utils import is_list_of_type

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@singledispatch
def combine_cols(*x: Any):
    """Combine 2-dimensional objects by row.

    If the first element in ``x`` contains a ``combine_cols`` method,
    the rest of the arguments are passed to that function.

    If all objects are :py:class:`~numpy.ndarray`, we use the
    :py:func:`~numpy.hstack` to combine dense matrices by row.

    If all objects are :py:class:`~scipy.sparse.spmatrix`, we use the
    :py:func:`~scipy.sparse.hstack` to combine sparse matrices by row.

    If the objects are a mix of sparse and dense matrices, we use the
    :py:func:`~scipy.sparse.hstack` function.

    Args:
        x (Any): Array of vector-like objects to combine.

            All elements of x are expected to be the same class or
            atleast compatible with each other.

    Raises:
        TypeError: If any object in the list cannot be coerced to a list.

    Returns:
        A combined matrix, typically the same type as the first element in ``x``.
        A :py:class:`~numpy.ndarray` if all elements are dense matrices.
        A :py:class:`~scipy.sparse.spmatrix`, if all elements are sparse
        If elements are a mix of dense and sparse, returns a
        :py:class:`~scipy.sparse.spmatrix` matrix.
    """

    first_object = x[0]

    if hasattr(first_object, "combine_cols"):
        return first_object.combine_cols(*x[1:])

    raise NotImplementedError("`combine_cols` method is not implement for objects.")


def _generic_numpy_scipy_combine_cols(*x: Any):
    from scipy.sparse import hstack

    return hstack(x)


@combine_cols.register(ndarray)
def _combine_cols_dense(*x: ndarray):
    if is_list_of_type(x, ndarray):
        from numpy import hstack

        return hstack(x)

    return _generic_numpy_scipy_combine_cols(*x)


@combine_cols.register(spmatrix)
def _combine_cols_sparse(*x: spmatrix):
    if is_list_of_type(x, spmatrix):
        from scipy.sparse import hstack

        return hstack(x)

    return _generic_numpy_scipy_combine_cols(*x)
