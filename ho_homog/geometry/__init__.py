# coding: utf8
"""
Created on 09/10/2018
@author: baptiste

Définition de classes d'objets géométriques et de fonctions permettant de créer un modèle géométrique de RVE dans gmsh.

sources :
    - https://deptinfo-ensip.univ-poitiers.fr/ENS/doku/doku.php/stu:python:pypoo

"""

import logging
import math
import warnings
from pathlib import Path

import gmsh
import matplotlib.pyplot as plt
import numpy as np

# nice shortcuts
model = gmsh.model
factory = model.occ

logger = logging.getLogger(__name__)


warnings.simplefilter("always")
# ? Doc: https://docs.python.org/3.6/library/warnings.html


def set_gmsh_option(option, val):
    """
    Set a gmsh option to the given value and print a log message.

    Parameters
    ----------
    option : string
    val : string or number
        Type of valid val depend of the option.
        See the gmsh reference manual for more information.

    """
    if isinstance(val, (int, float)):
        setter = gmsh.option.setNumber
        getter = gmsh.option.getNumber
    elif isinstance(val, str):
        setter = gmsh.option.setString
        getter = gmsh.option.getString
    else:
        raise TypeError("Wrong type of parameter for a gmsh option.")
    preval = getter(option)
    setter(option, val)
    logger.info(f"Gmsh option {option} set to {val} (previously : {preval}).")


from .curves import Curve, AbstractCurve, Arc, Line, bndry_logger
from .physical import PhysicalGroup
from .point import Point
from .surfaces import (
    AbstractSurface,
    LineLoop,
    PlaneSurface,
    surface_bool_cut,
    surface_bool_intersect,
)
from .tools import (
    angle_between,
    bisector,
    dual_base,
    macro_line_fragments,
    offset,
    remove_duplicates,
    round_corner,
    unit_vect,
)
from .transformations import plane_reflection, point_reflection, rotation, translation

DEFAULT_GMSH_OPTIONS = {
    "General.Terminal": 1,
    "General.Verbosity": 3,
    "Geometry.AutoCoherence": 0,
    "Mesh.ColorCarousel": 2,
    # * 0=by element type, 1=by elementary entity, 2=by physical entity,3=by partition
    "Mesh.MeshOnlyVisible": 0,
    # TODO : Should be in the init file of the mesh_tools module.
    "Mesh.CharacteristicLengthExtendFromBoundary": 0,
    "Mesh.SaveAll": 0,
    "Mesh.Binary": 0,
    "Mesh.MshFileVersion": 2.2,
    "Mesh.Algorithm": 5,
}


def init_geo_tools():
    """
    The Gmsh Python API must be initialized before using any functions.
    In addition, some options are set to custom values.
    """
    gmsh.initialize()  # ? Utiliser l'argument sys.argv ? cf script boolean.py
    for option, val in DEFAULT_GMSH_OPTIONS.items():
        set_gmsh_option(option, val)
    logger.info(f"gmsh module path : {Path(gmsh.__file__).resolve()}")
    logger.info("Gmsh SDK version : %s", gmsh.option.getString("General.Version"))


def reset():
    """Throw out all information about the created geometry and remove all gmsh models."""
    PhysicalGroup.all_groups = dict()
    gmsh.clear()


__all__ = [
    # * geometry global
    "logger",
    "set_gmsh_option",
    "init_geo_tools",
    "reset",
    # * points
    "Point",
    # * curves
    "Curve",
    "AbstractCurve",
    "Arc",
    "Line",
    "bndry_logger",
    # * physical entities
    "PhysicalGroup",
    # * surfaces
    "LineLoop",
    "PlaneSurface",
    "AbstractSurface",
    "surface_bool_cut",
    "surface_bool_intersect",
    # * tools
    "angle_between",
    "bisector",
    "dual_base",
    "macro_line_fragments",
    "offset",
    "remove_duplicates",
    "round_corner",
    "unit_vect",
    # * transformations
    "plane_reflection",
    "point_reflection",
    "rotation",
    "translation",
]
