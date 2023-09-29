from functools import singledispatch
from typing import List

__author__ = "jkanche"
__copyright__ = "jkanche"
__license__ = "MIT"


@singledispatch
def colnames(x) -> List[str]:
    """Access column names from 2-dimensional representations.

    Args:
        x: Any object.

    Raises:
        NotImplementedError: If ``x`` is not a supported type.

    Returns:
        List[str]: List of column names.
    """
    if hasattr(x, "colnames"):
        return x.colnames

    raise NotImplementedError(f"`colnames` is not supported for class: '{type(x)}'.")


@singledispatch
def set_colnames(x, names: List[str]):
    """Set column names.

    Args:
        x: Any object.
        names (List[str]): New names.

    Raises:
        NotImplementedError: if type is not supported.

    Returns:
        An object with the same type as ``x``.
    """
    raise NotImplementedError(
        f"`set_colnames` is not supported for class: '{type(x)}'."
    )
