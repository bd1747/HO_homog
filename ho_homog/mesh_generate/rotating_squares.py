#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 11:07:31 2021

@author: manon_thbaut
"""

import numpy as np
import logging
import ho_homog.geometry as geo
import gmsh
import ho_homog.mesh_tools as msh


from . import Gmsh2DRVE, logger

model = gmsh.model
factory = model.occ


def square_auxetics_RVE(
    theta, r, a=None, b=None, nb_cells=(1, 1), offset=(0.0, 0.0), name=""
):
    """[summary]

    Parameters
    ----------
    theta : float
        Angle d'ouverture, \in [0; \pi/4].
            0 -> configuration refermé
            \pi/4 -> configuration complètement ouverte
    r : float
        junction thinness, = rayon du cercle inscrit / côté d'un triangle
    a : float
        longueur du côté bas de la cellule
    b : float
        longueur des côtés des carrés constitutifs
    name : str


    Returns
    -------
    Instance of the Gmsh2DRVE class.

    Si a est imposée :
    La cellule est de taille fixe,
    par contre la longueur des côtés des carrés dépend de l'angle d'ouverture de la microstructure.
    Si b est imposé :
    C'est l'inverse.
    Carrés de dimensions fixes, dimensions de la cellules dépendent de l'ouverture.
    """

    logger.info("Start defining the geometry")
    name = name if name else "rot_square"

    if (not a) and (not b):
        raise ValueError("a or b (exclusive OR) must be imposed")
    elif a and b:
        raise ValueError("a and b cannot be imposed simultaneously")
    elif b:
        # si on choisit la taille des carrés b
        a = rotating_squares_square_2_cell(theta, b)
    elif a:
        b = rotating_squares_cell_2_square(theta, a)

    gen_vect = np.array(((a, 0.0), (0.0, a)))
    nb_cells, offset = np.asarray(nb_cells), np.asarray(offset)

    a0 = np.array((a, 0.0, 0.0))
    a1 = np.array((0.0, a, 0.0))

    s, c = np.sin(theta), np.cos(theta)

    vertices_base_ll = [
        [0, -b * c],
        [b * s, 0],
        [0, b * c],
        [-b * s, 0],
    ]
    translat_h = [
        (0.0, a / 2, 0.0),
        (a, a / 2, 0.0),
        (a / 2, 0.0, 0.0),
        (a / 2, a, 0.0),
    ]
    rot_1 = np.pi / 2

    translat_v = [(a / 2, a / 2, 0.0), (0.0, a, 0.0), (a, 0.0, 0.0), (a, a, 0.0)]
    # translat_v2 #
    z_axis = [0, 0, 1]
    # Definition of the LineLoops
    vertices_base_ll = [geo.Point(c) for c in vertices_base_ll]
    ll_base_vert = geo.LineLoop(vertices_base_ll, explicit=False)
    ll_base_horiz = geo.rotation(ll_base_vert, rot_1, z_axis)
    pattern_ll = [ll_base_vert]
    pattern_ll += [geo.translation(ll_base_vert, t) for t in translat_v]
    pattern_ll += [geo.translation(ll_base_horiz, t) for t in translat_h]
    pattern_ll = geo.remove_duplicates(pattern_ll)
    logger.info("Removing duplicate pattern line-loops: Done")
    logger.info(f"Number of pattern line-loops: {len(pattern_ll)}")
    for ll in pattern_ll:
        ll.round_corner_incircle(r)
    logger.info("Rounding all corners of pattern line-loops: Done")

    fine_pts = [pt for ll in pattern_ll for pt in ll.vertices]
    fine_pts = geo.remove_duplicates(fine_pts)

    return Gmsh2DRVE(pattern_ll, gen_vect, nb_cells, offset, fine_pts, False, name)


def rotating_squares_cell_2_square(theta, cell_lx):
    """Calcul de la longueur des côtés des carrés qui composent la microstructure.
    Formule de calcul valable uniquement pour la microstructure "rotating square".

    Parameters
    ----------
    theta : float
        angle, caractéristique de la configuration de la microstructure.
    cell_lx : float
        longueur du côté horizontal de la cellule unitaire
    """
    square_l = cell_lx / (2 * (np.cos(theta) + np.sin(theta)))
    return square_l


def rotating_squares_square_2_cell(theta, square_l):
    """Calcul de la longueur horizontale de la cellule unitaire à partir de la taille des carrés constitutifs.
    Formule de calcul valable uniquement pour la microstructure "rotating square".

    Parameters
    ----------
    theta : float
        angle, caractéristique de la configuration de la microstructure.
    square_l : float
        longueur des côtés des carrés qui composent la microstructure.
    """
    cell_lx = 2 * square_l * (np.cos(theta) + np.sin(theta))
    return cell_lx
