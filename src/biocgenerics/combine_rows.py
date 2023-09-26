from functools import singledispatch
from itertools import chain
from typing import Any

from numpy import ndarray
from scipy.sparse import spmatrix

from .utils import is_list_of_type

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@singledispatch
def combine_rows(*x: Any):
    """Combine 2-d objects by row.

    If the first element in ``x`` contains a ``combine_rows`` method,
    the rest of the arguments are passed to that function.

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

    if hasattr(first_object, "combine_rows"):
        return first_object.combine(*x[1:])

    raise NotImplementedError("`combin_rowse` method is not implement for objects.")


def _generic_numpy_scipy_combine_rows(*x: Any):
    from scipy.sparse import vstack
    return vstack(x)


@combine_rows.register(ndarray)
def _combine_rows_numpy(*x: ndarray):
    if is_list_of_type(x, ndarray):
        from numpy import vstack

        return vstack(x)
    
    return _generic_numpy_scipy_combine_rows(*x)


@combine_rows.register(spmatrix)
def _combine_rows_scipy(*x: spmatrix):
    if is_list_of_type(x, spmatrix):
        from scipy.sparse import vstack

        return vstack(x)
    
    return _generic_numpy_scipy_combine_rows(*x)
