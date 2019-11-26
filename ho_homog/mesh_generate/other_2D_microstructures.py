# coding: utf-8
"""
Created on 17/01/2019
@author: Baptiste Durand, baptiste.durand@enpc.fr
"""

import gmsh
import numpy as np
from more_itertools import flatten

import ho_homog.geometry as geo

from . import Gmsh2DRVE, logger

# nice shortcuts
model = gmsh.model
factory = model.occ


def auxetic_square_RVE(
    L, a, t, nb_cells=(1, 1), offset=(0.0, 0.0), soft_mat=False, name=""
):
    """
    Create an instance of Gmsh2DRVE that represents a RVE of
    the auxetic square microstructure.

    The generic operations will be performed with the Gmsh2DRVE methods.

    Parameters
    ----------
    L : float
        Length of the sides of the square cell.
    a : float
        Length of the slits beetween squares.
    t : float
        Width of the slits beetween squares.
    nb_cells : tuple or 1D array
        nb of cells in each direction of repetition
    offset : tuple or 1D array
        Relative position inside a cell of the point that
        will coincide with the origin of the global domain.

    Returns
    -------
    Instance of the Gmsh2DRVE class.
    """

    name = name if name else "aux_square"
    model.add(name)
    # geo.reset()

    offset = np.asarray(offset)
    nb_cells = np.asarray(nb_cells)

    logger.info("Start defining the auxetic_square geometry")
    gen_vect = np.array(((L, 0.0), (0.0, L)))
    b = (L - a) / 2.0
    e1 = np.array((L, 0.0, 0.0))
    e2 = np.array((0.0, L, 0.0))
    C = geo.Point(1 / 2.0 * (e1 + e2))
    M = geo.Point(1 / 4.0 * (e1 + e2))

    e3 = np.array((0.0, 0.0, 1.0))

    center_pts = [[(b, 0.0), (a + b, 0.0)], [(0.0, -a / 2.0), (0.0, a / 2.0)]]
    center_pts = [[geo.Point(np.array(coord)) for coord in gp] for gp in center_pts]
    center_lines = [geo.Line(*pts) for pts in center_pts]
    center_lines += [geo.point_reflection(ln, M) for ln in center_lines]
    center_lines += [geo.plane_reflection(ln, C, e1) for ln in center_lines]
    center_lines += [geo.plane_reflection(ln, C, e2) for ln in center_lines]
    center_lines = geo.remove_duplicates(center_lines)

    for ln in center_lines:
        ln.ortho_dir = np.cross(e3, ln.direction())
    pattern_ll = list()
    for ln in center_lines:
        vertices = [
            geo.translation(ln.def_pts[0], t / 2 * ln.ortho_dir),
            geo.translation(ln.def_pts[1], t / 2 * ln.ortho_dir),
            geo.translation(ln.def_pts[1], -t / 2 * ln.ortho_dir),
            geo.translation(ln.def_pts[0], -t / 2 * ln.ortho_dir),
        ]
        pattern_ll.append(geo.LineLoop(vertices))
    tmp_nb_bef = len(pattern_ll)
    pattern_ll = geo.remove_duplicates(pattern_ll)

    logger.debug(
        f"Number of line-loops removed from pattern-ll : {tmp_nb_bef - len(pattern_ll)}."
    )
    logger.debug(
        f"Final number of pattern line-loops for auxetic square : {len(pattern_ll)}"
    )

    for ll in pattern_ll:
        ll.round_corner_explicit(t / 2)
        filter_sides = list()
        # * Pour ne pas essayer d'ajouter au model gmsh des lignes de longueur nulle.
        # * (Error : could not create line)
        for crv in ll.sides:
            if not crv.def_pts[0] == crv.def_pts[-1]:
                filter_sides.append(crv)
        ll.sides = filter_sides

    fine_pts = geo.remove_duplicates(flatten([ln.def_pts for ln in center_lines]))

    return Gmsh2DRVE(pattern_ll, gen_vect, nb_cells, offset, fine_pts, soft_mat, name)
