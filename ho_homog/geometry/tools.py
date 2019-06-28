# coding: utf8
"""
Created on 24/06/2019
@author: baptiste

Geometry toolbox.
Perform diverse operations on numpy arrays, points or curves.

"""

from . import np, plt, logger, math
from .point import Point
from .curves import Line, Arc


import copy

E3 = np.array((0.0, 0.0, 1.0))


def unit_vect(v):
    """Normalize numpy arrays by length.

    If the input has a norm equal to zero, the same vector is returned.

    Parameters
    ----------
    v : numpy array
        1D numpy array

    Returns
    -------
    numpy array
    """
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def angle_between(v1, v2, orient=True):
    """Renvoie l'angle en radian, orienté ou non entre les deux vecteurs.

    Valeur comprise entre -pi (strictement) et pi.
    """
    # TODOC
    v1u = unit_vect(v1)
    v2u = unit_vect(v2)
    if orient:
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan2.html
        # ! EN 2D SEULEMENT
        angle = np.arctan2(v2u[1], v2u[0]) - np.arctan2(v1u[1], v1u[0])
        if angle <= -np.pi:
            angle += 2 * np.pi
        if angle > np.pi:
            angle -= 2 * np.pi
        return angle
    else:
        return np.arccos(np.clip(np.dot(v1u, v2u), -1.0, 1.0))


def bisector(u, v):
    """
    Compute the angle bisector.
    The angle is defined by two input vectors : u, v

    2D only for now.

    Returns
    -------
    numpy array
        unit vector that show the direction of the
    """

    u_, v_ = unit_vect(u), unit_vect(v)
    bis = np.cross(v_ - u_, E3)
    bis = unit_vect(bis)
    return bis


def dual_base(basis):
    """
    Calculates the dual basis associated with a given basis.

    Parameters
    ----------
    basis : numpy array_like, square matrix
        The components of the basis vectors in an orthogonal coordinate system.
        2-D array, Dimensions of the matrix : 2×2 or 3×3

    Return
    ------
    dual_b : np.array
        The components of the vectors that composed the dual base,
        in the same orthogonal coordinate system.

    """
    return np.linalg.inv(basis).T


def round_corner(inp_pt, pt_amt, pt_avl, r, junction_raduis=False, plot=False):
    """
    Calcul un arc de cercle nécessaire pour arrondir un angle défini par trois points.

    Le rayon de l'arc peut être contrôlé explicitement ou
    par la méthode du cercle inscrit.
    Deux objets de type Line sont aussi renvoyés.
    Ces segments orientés relient les deux points en amont et aval de l'angle à l'arc.

    """
    from .transformations import translation
    # Direction en amont et en aval
    v_amt = unit_vect(pt_amt.coord - inp_pt.coord)
    v_avl = unit_vect(pt_avl.coord - inp_pt.coord)

    alpha = angle_between(v_amt, v_avl, orient=True)
    v_biss = bisector(v_amt, v_avl)
    if alpha < 0:
        # corriger le cas des angles \in ]-pi;0[ = ]pi;2pi[]. L'arrondi est toujours dans le secteur <180° #noqa
        v_biss = -v_biss

    if junction_raduis:
        if plot:
            R = copy.deepcopy(r)
        r = (
            r
            * abs(math.tan(alpha / 2.0))
            * abs(math.tan(abs(alpha / 4.0) + math.pi / 4.0))
        )

    # Calcul des distances centre - sommet de l'angle et sommet - point de raccordement
    dist_center = float(r) / abs(math.sin(alpha / 2.0))
    dist_racc = float(r) / abs(math.tan(alpha / 2.0))

    pt_racc_amt = translation(inp_pt, dist_racc * v_amt)
    pt_racc_avl = translation(inp_pt, dist_racc * v_avl)
    pt_ctr = translation(inp_pt, dist_center * v_biss)

    round_arc = Arc(pt_racc_amt, pt_ctr, pt_racc_avl)
    racc_amt = Line(pt_amt, pt_racc_amt)
    racc_avl = Line(pt_racc_avl, pt_avl)
    geoList = [racc_amt, round_arc, racc_avl]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if junction_raduis:
            ax.add_patch(plt.Circle(inp_pt.coord, R, color="orange", fill=True))
        pts = [inp_pt, pt_amt, pt_avl]
        colors = ["black", "red", "blue"]
        x, y = list(), list()
        for pt, color in zip(pts, colors):
            pt.plot2D(color)
            x.append(pt.coord[0])
            y.append(pt.coord[1])
        pt_racc_amt.plot2D("black")
        pt_racc_avl.plot2D("black")
        pt_ctr.plot2D("purple")
        racc_amt.plot2D("red")
        racc_avl.plot2D("blue")
        ax.add_patch(round_arc.plot2D("purple"))
        plt.axis("equal")
        plt.xlim(min(x), max(x))
        plt.ylim(min(y), max(y))
        plt.show()

    return geoList


def offset(pt, pt_dir1, pt_dir2, t):
    # TODOC
    v1 = pt_dir1.coord - pt.coord
    v2 = pt_dir2.coord - pt.coord
    alpha = angle_between(v1, v2, orient=False)
    v_biss = bisector(v1, v2)
    if alpha != np.pi:
        dpcmt = -t / abs(np.sin(alpha / 2.0))
    else:
        dpcmt = -t
    new_coord = pt.coord + dpcmt * v_biss
    return Point(new_coord)


def macro_line_fragments(curves, main_line):
    """
    Extract from a set of curves those which represent a part of the main line.

    Designed to identify in the list of the boundary elements of the microstructure
    those that are on a given border of the RVE.

    Parameters
    ----------
    curves: list of instances of a Curve subclass
    main_line : instance of Curve subclass

    """

    for ln in curves + [main_line]:
        if not ln.tag:
            ln.add_gmsh()
            logger.debug(
                f"In gather_boundary_fragments, curve {ln.tag} added to the model"
            )
        if not ln.def_pts:
            ln.get_boundary()
    main_start = main_line.def_pts[0].coord
    main_dir = main_line.def_pts[-1].coord - main_line.def_pts[0].coord
    parts = list()
    for crv in curves:
        crv_dir = crv.def_pts[-1].coord - crv.def_pts[0].coord
        if not np.allclose([0, 0, 0], np.cross(main_dir, crv_dir), rtol=0, atol=1e-08):
            continue
        mix_dir = 1 / 2 * (crv.def_pts[-1].coord + crv.def_pts[0].coord) - main_start
        if not np.allclose([0, 0, 0], np.cross(main_dir, mix_dir), rtol=0, atol=1e-08):
            continue
        parts.append(crv)
    return parts


def remove_duplicates(ent_list):
    """
    Remove all duplicates from a list of geometrical entities.

    Designed to be used with instances of one of the geometrical classes
    (Point, Curve, LineLoop and Surface).
    It should be noted that the equality operator has been overidden for these classes.
    For the LineLoop instances, it takes into account the lineloop direction
    (clockwise/anticlockwise).

    Note
    --------
    Since the __eq__ method has been overridden in the definition
    of the geometrical classes, their instances are not hashable.
    Faster solution for hashable objects :
    https://stackoverflow.com/questions/7961363/removing-duplicates-in-lists/7961425#7961425 #noqa

    """
    unique = list()
    for x in ent_list:
        for y in unique:
            if x == y:
                break
        else:
            unique.append(x)
    return unique
