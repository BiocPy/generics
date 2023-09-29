from functools import singledispatch
from itertools import chain
from typing import Any

from numpy import concatenate, ndarray
from scipy.sparse import sparray, hstack

from .utils import is_list_of_type

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


def _generic_combine(*x: Any):
    try:
        _all_as_list = [list(m) for m in x]
        return list(chain(*_all_as_list))
    except Exception as e:
        raise NotImplementedError(
            "`combine` method is not implement for objects."
        ) from e


@singledispatch
def combine(*x: Any):
    """Combine vector-like objects.

    Custom classes may implement the ``combine`` method.

    If the first element in ``x`` contains a ``combine`` method,
    the rest of the arguments are passed to that function.

    If all objects are either a :py:class:`~numpy.ndarray`, or
    :py:class:`~numpy.array`, we combine all arrays using numpy's
    :py:func:`~numpy.concatenate`.

    If all objects are a :py:class:`~scipy.sparse.spmatrix`,
    these objects are combined using scipy's :py:class:`~scipy.sparse.hstack`.

    For all other scenario's, all objects are coerced to alist and combined.

    Args:
        x (Any): Array of vector-like objects to combine.

            All elements of x are expected to be the same class or
            atleast compatible with each other.

    Raises:
        TypeError: If any object in the list cannot be coerced to a list.

    Returns:
        A combined object, typically the same type as the first element in ``x``.
        A list if one of the objects is a list.
    """

    first_object = x[0]

    if hasattr(first_object, "combine"):
        return first_object.combine(*x[1:])

    return _generic_combine(*x)


@combine.register(ndarray)
def _combine_dense_arrays(*x: ndarray):
    if is_list_of_type(x, ndarray):
        return concatenate(x)

    return _generic_combine(*x)


@combine.register(sparray)
def _combine_sparse_arrays(*x: sparray):
    if is_list_of_type(x, sparray):
        return hstack(x)

    return _generic_combine(*x)