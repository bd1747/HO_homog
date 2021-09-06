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


def square_auxetics_RVE(theta, r, a=None, b=None, nb_cells=(1, 1), offset=(0.0, 0.0), name=""):
    """[summary]

    Parameters
    ----------
    theta : float
        Paramètre d'ouverture, intervalle [0, pi/4]
        0: configuration refermé
        pi/4: configuration complètement ouverte
    name : str

    r : float
        junction thinness, rayon de jonction / côté d'un triangle

    Returns
    -------
    Instance of the Gmsh2DRVE class.

    La cellule est de taille constante,
    par contre les carrés ont une taille qui dépend de l'angle d'ouverture.
    """

    logger.info("Start defining the geometry")
    name = name if name else "kagome"

    if (not a) and (not b):
        raise ValueError("a or b (exclusive OR) must be imposed")
    elif a and b:
        raise ValueError("a and b cannot be imposed simultaneously")
    elif b:
        # si on choisit la taille des triangles b
        a = 2*b*(np.cos(theta)+np.sin(theta))
    elif a:
        b = a/(2*(np.cos(theta)+np.sin(theta)))

    gen_vect = np.array(((a, 0.0), (0.0, a)))
    nb_cells, offset = np.asarray(nb_cells), np.asarray(offset)

    a0 = np.array((a, 0.0, 0.0))
    a1 = np.array((0.0, a, 0.0))
   
    s=np.sin(theta)
    c=np.cos(theta)
    
    vertices_base_ll = [
            [0,-b*c],
            [b*s,0],
            [0,b*c],
            [-b*s,0],
            ]
    translat_h = [(0.,a/2,0.), (a,a/2,0.), (a/2,0.,0.), (a/2,a,0.)]
    rot_1 = np.pi/2
    
    translat_v = [(a/2, a/2, 0.), (0.0, a, 0.), (a, 0., 0.), (a, a, 0.)]
    # translat_v2 #
    z_axis = [0, 0, 1]
    ll_base_vert = geo.LineLoop([geo.Point(c) for c in vertices_base_ll],explicit=False)
    ll_base_horiz = geo.rotation(ll_base_vert,rot_1,z_axis)
    
    pattern_ll = [ll_base_vert]
    pattern_ll += [geo.translation(ll_base_vert, t) for t in translat_v]
    pattern_ll += [geo.translation(ll_base_horiz, t) for t in translat_h]
    
    for ll in pattern_ll:
        ll.round_corner_incircle(r)
        
    pattern_ll = geo.remove_duplicates(pattern_ll)
    logger.info("Removing duplicate pattern line-loops: Done")
    logger.info(f"Number of pattern line-loops: {len(pattern_ll)}")
    for ll in pattern_ll:
        ll.round_corner_incircle(r)
     
    logger.info("Rounding all corners of pattern line-loops: Done")
    fine_pts = [pt for ll in pattern_ll for pt in ll.vertices]
    fine_pts = geo.remove_duplicates(fine_pts)

    return Gmsh2DRVE(pattern_ll, gen_vect, nb_cells, offset, fine_pts, False, name,)