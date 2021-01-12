#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:12:32 2020

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

# * Logging
logger = logging.getLogger(__name__)  # http://sametmax.com/ecrire-des-logs-en-python/
logger.setLevel(logging.INFO)


formatter = logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s", "%H:%M")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

xdmf_parameters = dict(
    flush_output=False, functions_share_mesh=True, rewrite_function_mesh=False
)


def kagome_RVE(alpha, r, a=None, b=None, nb_cells=(1, 1), offset=(0.0, 0.0), name=""):
    """[summary]

    Parameters
    ----------
    alpha : float
        Paramètre d'ouverture, intervalle [0, 0.5]
        0: configuration refermé
        0.5: configuration complétement ouverte
    name : str

    r : float
        junction thinness, rayon de jonction / côté d'un triangle

    Returns
    -------
    Instance of the Gmsh2DRVE class.

    La cellule est de taille constante,
    par contre les triangles ont une taille qui dépend de l'angle d'ouverture.
    """

    logger.info("Start defining the geometry")
    name = name if name else "kagome"

    if (not a) and (not b):
        raise ValueError("a or b (exclusive OR) must be imposed")
    elif a and b:
        raise ValueError("a and b cannot be imposed simultaneously")
    elif b:
        # si on choisit la taille des triangles b
        a = kagome_triangle_size_2_cell_size(alpha, b)
    elif a:
        b = kagome_cell_size_2_triangle_size(alpha, a)

    gen_vect = np.array(((a, a / 2), (0.0, np.sqrt(3) / 2 * a)))
    nb_cells, offset = np.asarray(nb_cells), np.asarray(offset)

    a0 = np.array((a, 0.0, 0.0))
    a1 = np.array((a / 2, np.sqrt(3) / 2 * a, 0.0))
    phi1 = geo.Point(alpha * a0)
    psi1 = geo.Point((1 - alpha) * a1)
    z_axis = [0, 0, 1]
    angle = 2 * np.pi / 3
    phi2 = geo.rotation(phi1, angle, z_axis)
    psi2 = geo.rotation(psi1, angle, z_axis)
    phi3 = geo.rotation(phi2, angle, z_axis)
    psi3 = geo.rotation(psi2, angle, z_axis)

    star_pts = [phi1, psi1, phi2, psi2, phi3, psi3]
    pattern_ll = [geo.LineLoop(star_pts, explicit=False)]
    trans = [a0, a1, a0 + a1]
    pattern_ll += [geo.translation(pattern_ll[0], t) for t in trans]

    pattern_ll = geo.remove_duplicates(pattern_ll)
    logger.info("Removing duplicate pattern line-loops: Done")
    logger.info(f"Number of pattern line-loops: {len(pattern_ll)}")
    for ll in pattern_ll:
        ll.round_corner_kagome(r * b, a, alpha)
    logger.info("Rounding all corners of pattern line-loops: Done")
    fine_pts = [pt for ll in pattern_ll for pt in ll.vertices]
    fine_pts = geo.remove_duplicates(fine_pts)

    return Gmsh2DRVE(pattern_ll, gen_vect, nb_cells, offset, fine_pts, False, name,)



def kagome_triangle_size_2_cell_size(alpha, b):
    """Passage de b (côté des triangles) à a (taille caractéristique cellule unitaire du kagome)."""
    _a = 1.0  # pour cellule de hauteur 1
    x = (1 - alpha) * _a / 2 - alpha * _a
    y = (1 - alpha) * _a * np.sqrt(3) / 2
    _b = np.sqrt(x ** 2 + y ** 2)
    theta = np.arcsin(np.sqrt(3) * alpha / (2 * _b))
    # compute theta for a unit cell

    a = 2 * b * np.cos(np.pi / 3 - theta)  # real cell
    return a


def kagome_cell_size_2_triangle_size(alpha, a):
    """Pour la microstructure "kagome", passage de a (taille caractéristique de la cellule unitaire) à  b (côté des triangles)

    Parameters
    ----------
    alpha : float
        paramètre d'ouverture de la microstructure
    a : float
        taille cellule unitaire parallélogramme.
    """
    t1 = (1 - alpha) * a / 2 - alpha * a
    t2 = (1 - alpha) * np.sqrt(3) * a / 2
    b = np.sqrt(t1 ** 2 + t2 ** 2)
    return b
