# coding: utf-8
"""
Created on 17/01/2019
@author: Baptiste Durand, baptiste.durand@enpc.fr
"""

import copy
import gmsh
import numpy as np

import ho_homog.geometry as geo

from . import Gmsh2DRVE, logger

# nice shortcuts
model = gmsh.model
factory = model.occ


def pantograph_RVE(
    a, b, k, junction_r, nb_cells=(1, 1), offset=(0.0, 0.0), soft_mat=False, name=""
):
    """
    Create an instance of Gmsh2DRVE that represents a RVE of the pantograph microstructure.
    The geometry that is specific to this microstructure is defined in this staticmethod. Then, the generic operations will be performed with the Gmsh2DRVE methods.

    Parameters
    ----------
    a,b and k : floats
        main lengths of the microstruture
    junction_r : float
        radius of the junctions inside the pantograph microstructure
    nb_cells : tuple or 1D array
        nb of cells in each direction of repetition
    offset : tuple or 1D array
        Relative position inside a cell of the point that will coincide with the origin of the global domain

    Returns
    -------
    Instance of the Gmsh2DRVE class.
    """
    name = name if name else "pantograph"

    offset = np.asarray(offset)
    nb_cells = np.asarray(nb_cells)

    logger.info("Start defining the pantograph geometry")
    Lx = 4 * a
    Ly = 6 * a + 2 * b
    cell_vect = np.array(((Lx, 0.0), (0.0, Ly)))

    e1 = np.array((a, 0.0, 0.0))
    e2 = np.array((0.0, a, 0.0))
    p = np.array((k, 0.0, 0.0))
    b_ = b / a * e2
    E1 = geo.Point(e1)
    E2 = geo.Point(e2)
    E1m = geo.Point(-1 * e1)
    E2m = geo.Point(-1 * e2)
    L = geo.Point(2 * (e1 + e2))
    Lm = geo.Point(2 * (e1 - e2))
    M = geo.Point(e1 + 1.5 * e2 + b_ / 2)
    I = geo.Point(2 * (e1 + 1.5 * e2 + b_ / 2))

    contours = list()
    contours.append([E1, E2, E1m, E2m])
    contours.append([E1, Lm, geo.Point(3 * e1), L])
    contours.append(
        [
            E2,
            L,
            geo.translation(L, 0.5 * b_ - p),
            geo.translation(L, b_),
            geo.translation(E2, b_),
            geo.translation(E2, 0.5 * b_ + p),
        ]
    )
    pattern_ll = [geo.LineLoop(pt_list, explicit=False) for pt_list in contours]
    pattern_ll += [geo.point_reflection(ll, M) for ll in pattern_ll]
    sym_ll = [geo.plane_reflection(ll, I, e1) for ll in pattern_ll]
    for ll in sym_ll:
        ll.reverse()
    pattern_ll += sym_ll
    sym_ll = [geo.plane_reflection(ll, I, e2) for ll in pattern_ll]
    for ll in sym_ll:
        ll.reverse()

    pattern_ll += sym_ll
    pattern_ll = geo.remove_duplicates(pattern_ll)
    logger.info("Done removing of the line-loops duplicates")
    for ll in pattern_ll:
        ll.round_corner_incircle(junction_r)
    logger.info("Done rounding all corners of pattern line-loops")

    constr_pts = [pt for ll in pattern_ll for pt in ll.vertices]
    fine_pts = [
        pt
        for pt in constr_pts
        if (pt.coord[0] % 1 < p[0] / 2.0 or pt.coord[0] % 1 > 1.0 - p[0] / 2.0)
    ]
    fine_pts = geo.remove_duplicates(fine_pts)
    return Gmsh2DRVE(pattern_ll, cell_vect, nb_cells, offset, fine_pts, soft_mat, name)


def pantograph_offset_RVE(
    a,
    b,
    k,
    thickness,
    fillet_r=0.0,
    nb_cells=(1, 1),
    offset=(0.0, 0.0),
    soft_mat=False,
    name="",
):
    """
    Generate a RVE object for the pantograph microstructure.
    Junctions are obtained by creating offset curves from the microstructure contour.

    Parameters
    ----------
    a,b and k : floats
        main lengths of the microstruture
    thickness : float
        distance of translation prescribed for the vertices of the contour
    fillet_r : float
        radius of the fillets for the contour
    nb_cells : tuple or 1D array
        nb of cells in each direction of repetition
    offset : tuple or 1D array
        Relative position inside a cell of the point that will coincide with the origin of the global domain

    Returns
    -------
    Instance of the Gmsh2DRVE class.
    """

    name = name if name else "pantograph"

    offset = np.asarray(offset)
    nb_cells = np.asarray(nb_cells)

    logger.info("Start defining the pantograph geometry")

    Lx = 4 * a
    Ly = 6 * a + 2 * b
    cell_vect = np.array(((Lx, 0.0), (0.0, Ly)))

    e1 = np.array((a, 0.0, 0.0))
    e2 = np.array((0.0, a, 0.0))
    p = np.array((k, 0.0, 0.0))
    b_ = b / a * e2
    E1 = geo.Point(e1)
    E2 = geo.Point(e2)
    E1m = geo.Point(-1 * e1)
    E2m = geo.Point(-1 * e2)
    L = geo.Point(2 * (e1 + e2))
    Lm = geo.Point(2 * (e1 - e2))
    M = geo.Point(e1 + 1.5 * e2 + b_ / 2)
    I = geo.Point(2 * (e1 + 1.5 * e2 + b_ / 2))
    contours = list()
    contours.append([E1, E2, E1m, E2m])
    contours.append([E1, Lm, geo.Point(3 * e1), L])
    contours.append(
        [
            E2,
            L,
            geo.translation(L, 0.5 * b_ - p),
            geo.translation(L, b_),
            geo.translation(E2, b_),
            geo.translation(E2, 0.5 * b_ + p),
        ]
    )
    pattern_ll = [geo.LineLoop(pt_list, explicit=False) for pt_list in contours]
    pattern_ll += [geo.point_reflection(ll, M) for ll in pattern_ll]
    sym_ll = [geo.plane_reflection(ll, I, e1) for ll in pattern_ll]
    for ll in sym_ll:
        ll.reverse()
    pattern_ll += sym_ll
    sym_ll = [geo.plane_reflection(ll, I, e2) for ll in pattern_ll]
    for ll in sym_ll:
        ll.reverse()
    pattern_ll += sym_ll
    pattern_ll = geo.remove_duplicates(pattern_ll)
    logger.info("Done removing of the line-loops duplicates")

    constr_pts = [copy.deepcopy(pt) for ll in pattern_ll for pt in ll.vertices]

    for ll in pattern_ll:
        ll.offset(thickness)

    if fillet_r:
        for ll in pattern_ll:
            ll.round_corner_explicit(fillet_r)
        logger.info("Done rounding all corners of pattern line-loops")

    fine_pts = [
        pt
        for pt in constr_pts
        if (pt.coord[0] % 1 < p[0] / 2.0 or pt.coord[0] % 1 > 1.0 - p[0] / 2.0)
    ]
    fine_pts = geo.remove_duplicates(fine_pts)
    return Gmsh2DRVE(pattern_ll, cell_vect, nb_cells, offset, fine_pts, soft_mat, name)


def beam_pantograph_RVE(
    a, b, w, junction_r=0.0, nb_cells=(1, 1), offset=(0.0, 0.0), soft_mat=False, name=""
):
    """
    Create an instance of Gmsh2DRVE that represents a RVE of the beam pantograph microstructure.
    The geometry that is specific to this microstructure is defined in this staticmethod. Then, the generic operations will be performed with the Gmsh2DRVE methods.

    Parameters
    ----------
    a, b : floats
        main lengths of the microstruture
    w : float
        width of the constitutive beams
    junction_r : float, optional
        Radius of the corners/fillets that are created between concurrent borders of beams.
        The default is 0., which implies that the angles will not be rounded.
    nb_cells : tuple or 1D array, optional
        nb of cells in each direction of repetition (the default is (1, 1).)
    offset : tuple, optional
        If (0., 0.) or False : No shift of the microstructure.
        Else : The microstructure is shift with respect to the macroscopic domain.
        offset is the relative position inside a cell of the point that will coincide with the origin of the global domain.
    soft_mat : bool, optional
        If True : the remaining surface inside the RVE is associated with a second material domain and a mesh is genereted to represent it.
        Else, this space remains empty.
    name : str, optional
        The name of the RVE. It is use for the gmsh model and the mesh files.
        If name is '' (default) or False, the name of the RVE is 'beam_pantograph'.

    Returns
    -------
    Instance of the Gmsh2DRVE class.
    """

    name = name if name else "beam_pantograph"
    offset = np.asarray(offset)
    nb_cells = np.asarray(nb_cells)

    logger.info("Start defining the beam pantograph geometry")
    Lx = 4 * a
    Ly = 6 * a + 2 * b
    cell_vect = np.array(((Lx, 0.0), (0.0, Ly)))
    e1 = np.array((a, 0.0, 0.0))
    e2 = np.array((0.0, a, 0.0))
    b_ = b / a * e2
    E1 = geo.Point(e1)
    E2 = geo.Point(e2)
    E1m = geo.Point(-1 * e1)
    E2m = geo.Point(-1 * e2)
    L = geo.Point(2 * (e1 + e2))
    Lm = geo.Point(2 * (e1 - e2))
    M = geo.Point(e1 + 1.5 * e2 + b_ / 2)
    I = geo.Point(2 * (e1 + 1.5 * e2 + b_ / 2))
    contours = [
        [E1, E2, E1m, E2m],
        [E1, Lm, geo.Point(3 * e1), L],
        [E1, L, E2],
        [E2, L, geo.translation(L, b_), geo.translation(E2, b_)],
    ]
    pattern_ll = [geo.LineLoop(pt_list, explicit=False) for pt_list in contours]
    pattern_ll += [geo.point_reflection(ll, M) for ll in pattern_ll]
    sym_ll = [geo.plane_reflection(ll, I, e1) for ll in pattern_ll]
    for ll in sym_ll:
        ll.reverse()
    pattern_ll += sym_ll
    sym_ll = [geo.plane_reflection(ll, I, e2) for ll in pattern_ll]
    for ll in sym_ll:
        ll.reverse()
    pattern_ll += sym_ll
    pattern_ll = geo.remove_duplicates(pattern_ll)
    constr_pts = [copy.deepcopy(pt) for ll in pattern_ll for pt in ll.vertices]
    for ll in pattern_ll:
        ll.offset(w)
    if junction_r:
        for ll in pattern_ll:
            ll.round_corner_incircle(junction_r)

    fine_pts = geo.remove_duplicates(constr_pts)
    return Gmsh2DRVE(pattern_ll, cell_vect, nb_cells, offset, fine_pts, soft_mat, name)


def pantograph_E11only_RVE(
    a,
    thickness,
    fillet_r=0.0,
    nb_cells=(1, 1),
    offset=(0.0, 0.0),
    soft_mat=False,
    name="",
):
    """
    Generate a RVE object for the simplified pantograph microstructure.
    Only one floppy mode : E11
    Junctions are obtained by creating offset curves from the microstructure contour.
    + fillets -> avoid stress concentration

    Parameters
    ----------
    a: floats
        main dimension of the microstruture
    thickness : float
        distance of translation prescribed for the vertices of the contour
    fillet_r : float
        radius of the fillets for the contour
    nb_cells : tuple or 1D array
        nb of cells in each direction of repetition
    offset : tuple or 1D array
        Relative position inside a cell of the point that will coincide with the origin of the global domain

    Returns
    -------
    Instance of the Gmsh2DRVE class.
    """
    name = name if name else "pantograph"
    offset = np.asarray(offset)
    nb_cells = np.asarray(nb_cells)
    logger.info("Start defining the pantograph geometry")
    Lx = 4 * a
    Ly = 4 * a + 2 * thickness
    cell_vect = np.array(((Lx, 0.0), (0.0, Ly)))
    e1 = np.array((a, 0.0, 0.0))
    e2 = np.array((0.0, a, 0.0))
    pt_L = 2 * (e1 + e2)

    square = [e1, e2, -e1, -e2]
    square = [p + pt_L for p in square]
    square = [geo.Point(p) for p in square]

    rhombus_v = [np.array((0.0, 0.0)), e1 + 2 * e2, 4 * e2, -e1 + 2 * e2]
    rhombus_v = [geo.Point(p) for p in rhombus_v]

    cut_shape_h = [
        np.array((0.0, 0.0)),
        0.25 * (-2 * e1 + e2),
        0.25 * (-4 * e1 + e2),
        0.25 * (-4 * e1 - e2),
        4 * e1 + 0.25 * (4 * e1 - e2),
        4 * e1 + 0.25 * (4 * e1 + e2),
        4 * e1 + 0.25 * (2 * e1 + e2),
        4 * e1,
        e2 + 2 * e1,
    ]
    cut_shape_h = [geo.Point(p) for p in cut_shape_h]
    
    square = geo.LineLoop(square, explicit=False)
    rhombus_v = geo.LineLoop(rhombus_v, explicit=False)
    cut_shape_h = geo.LineLoop(cut_shape_h, explicit=False)
    pattern = [square, rhombus_v, cut_shape_h]
    sym_rhombus = [
        geo.plane_reflection(rhombus_v, pt_L, e1),
        geo.plane_reflection(cut_shape_h, pt_L, e2),
    ]
    for ll in sym_rhombus:
        ll.reverse()
    pattern += sym_rhombus
    pattern = geo.remove_duplicates(pattern)
    translated_pattern = list()
    for ll in pattern:
        translated_pattern.append(geo.translation(ll, np.array((0.0, thickness, 0.0))))
    logger.info("Done removing of the line-loops duplicates")
    pattern = translated_pattern
    constr_pts = [copy.deepcopy(pt) for ll in pattern for pt in iter((ll.vertices))]
    for ll in pattern:
        ll.offset(thickness)

    if fillet_r:
        for ll in pattern:
            ll.round_corner_explicit(fillet_r)
    logger.info("Done rounding all corners of pattern line-loops")
    fine_pts = geo.remove_duplicates(constr_pts)
    return Gmsh2DRVE(pattern, cell_vect, nb_cells, offset, fine_pts, soft_mat, name)
