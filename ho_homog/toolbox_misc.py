# coding: utf-8
"""
Created on 19/02/2021
@author: Baptiste Durand, baptiste.durand@enpc.fr

Collection of miscellaneous tools used in HO_homog.

"""
import logging

logger = logging.getLogger(__name__)


def _wrap_in_list(obj, name, types=type):
    """
    Transform single object or a collection of objects into a list.

    Source
    ------
    python/dolfin/fem/assembling.py, commit 4c72333
    """

    if obj is None:
        lst = []
    elif hasattr(obj, "__iter__"):
        lst = list(obj)
    else:
        lst = [obj]
    for obj in lst:
        if not isinstance(obj, types):
            raise TypeError(f"expected a (list of) {types} as {name} argument")
    return lst
