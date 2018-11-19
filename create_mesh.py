# -*- coding: utf-8 -*-
"""
Created on 17/11/2018
@author: baptiste

"""

import geometry as geo
import mesh_tools as msh
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import gmsh
import os
import matplotlib.pyplot as plt
import math

# nice shortcuts
model = gmsh.model
factory = model.occ

logger = logging.getLogger() #http://sametmax.com/ecrire-des-logs-en-python/
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s') # Afficher le temps à chaque message
file_handler = RotatingFileHandler('activity.log', 'a', 1000000, 1)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler) #Pour écriture d'un fichier log
formatter = logging.Formatter('%(levelname)s :: %(message)s') 
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler) #Pour écriture du log dans la console

def duplicate_pattern(cell_ll, nb_cells, gen_vect):
    """
    Propager la cellule de base dans selon les directions de l'espace.

    Parameters
    ----------
    cell_ll : list
        The LineLoops that will be replicated. They define the pattern for a unit cell.
    np_cells : tuple, dimension 2 or 3
        Number of cells in each direction.
    gen_vect : array
        The generating vectors that are related to the given microstruture.
    
    Returns
    -------
    repeated_ll : list
        Repeated lineloops that define the pattern over the whole domain associated with the given numbers of cells.
    """
    repeated_ll = cell_ll
    for k in range(len(nb_cells)):
        if nb_cells[k] > 1 :
            new_contours = list()
            for i in range(1, nb_cells[k]):
                new_contours += [geo.translation(ll, i*gen_vect[k]) for ll in repeated_ll]
            repeated_ll += new_contours
    geo.remove_duplicates(repeated_ll)
    return repeated_ll

def offset_pattern(cell_ll, offset_vect, gen_vect):
    t_vect = -1 * np.array([val % gen_vect[i][i] for i,val in enumerate(offset_vect)])
        #GeneratingVector[i][i] pour récupérer la longueur de la cellule selon la direction i.
        #L'opératation modulo avec des floats est possible. Résultat du signe du second terme.
    shifted_ll = [geo.translation(ll, t_vect) for ll in cell_ll] geo.translation(cell_ll, t_vect)
    return shifted_ll

class GeoAndMesh(object):
    """
    Contrat :
    Pour un certain choix de microstructure, construit un mesh qui représente un milieu périodique sur une certaine étendue.

    Une instance contient :
    name : string
        le nom du maillage ou le chemin complet vers le maillage
    physical_groups : dictionnary
        a dictionnary that catalogs all the physical groups that have been created and exported in the msh file.
    gen_vect : square array, 2D or 3D
        The generating vectors that are related to the given microstruture.

    """
    def __init__(self,):
        self.mesh_path = #? ou name ?
        self.physical_groups =


    @staticmethod
    def pantograph(cell_dim, junction_r, nb_cells=(1, 1), offset=(0., 0.), soft_mat=False, name=''):
        """
        Contrat :
        Pour un certain choix de microstructure, construit un mesh qui représente un milieu périodique sur une certaine étendue. 
        Parameters
        ----------
        cell_dim : dictionnary
            main length of the microstruture
        junction_r : float
            radius of the junctions inside the pantograph microstructure
        nb_cells : tuple
            nb of cells in each direction of repetition
        offset : tuple or array
            Relative position inside a cell of the point that will coincide with the origin of the global domain
        Returns
        -------
        Instance of the GeoAndMesh class.
        """

        name = name if name else "pantograph" 
        model.add(name)
        geo.reset()

        logger.info('Start defining the pantograph geometry')
        a = cell_dim['a']
        b = cell_dim['b']
        k = cell_dim['k']
        r = junction_r

        Lx = 4*a
        Ly = 6*a+2*b
        gen_vect = np.array(((Lx,0.), (0.,Ly)))

        e1 = np.array((a, 0., 0.))
        e2 = np.array((0., a, 0.))
        p = np.array((k, 0., 0.))
        b = b/a*e2
        E1 = geo.Point(e1)
        E2 = geo.Point(e2)
        E1m = geo.Point(-1*e1)
        E2m = geo.Point(-1*e2)
        O = np.zeros((3,))
        L = geo.Point(2*(e1+e2))
        Lm = geo.Point(2*(e1-e2))
        M = geo.Point(e1 + 1.5*e2 + b/2)
        I = geo.Point(2*(e1 + 1.5*e2 + b/2))

        contours = list()
        contours.append([E1, E2, E1m, E2m])
        contours.append([E1, Lm, geo.Point(3*e1), L])
        contours.append([E2, L, geo.translation(L,b/2-p), geo.translation(L,b), geo.translation(E2,b), geo.translation(E2,b/2+p)])
        pattern_ll = [geo.LineLoop(pt_list, explicit=False) for pt_list in contours]

        #? Créer une méthode statique pour réaliser ces opérations de symétries. Parameters : la liste de lineloop et une liste qui indique les opérations de symétries
        pattern_ll += [geo.point_reflection(ll, M) for ll in pattern_ll]
        pattern_ll += [geo.plane_reflection(ll, I, e1) for ll in pattern_ll]
        pattern_ll += [geo.plane_reflection(ll, I, e2) for ll in pattern_ll]
        geo.remove_duplicates(pattern_ll)
        logger.info('Done removing of the line-loops duplicates')
#! REPRENDRE ICI
        #? Idem, utiliser un array pour offset ?
        if np.asarray(offset).any():
            effect_nb_cells = [int(math.ceil(val+1)) if offset_v[i] != 0 else int(math.ceil(val)) for i,val in enumerate(nb_cells)]
            pattern_ll = offset_cell(pattern_ll, offset_v, gener_vect)
        else:
            nb_cells_ = [int(math.ceil(val)) for val in nb_cells]
   
        #? Utiliser un array pour nb_cells ?
        if not np.equal(np.asarray(nb_cells),1).all():
            duplicate_pattern(pattern_ll, nb_cells_, gen_vect)
        
        for ll in pattern_ll:
            ll.round_corner_incircle(r)
        logger.info('Done rounding all corners of pattern line-loops')

        macro_vtcs = [O, gen_vect[0], gen_vect[0] + gen_vect[1] , gen_vect[1]] #TODO là il faut modifier pour prendre en compte le nb de cellules. Ce n'est pas gen_vect mais un multiple.
        macro_ll = geo.LineLoop([geo.Point(c) for c in macro_vtcs])
        macro_s = geo.PlaneSurface(macro_ll)

        logger.info('Start boolean operations on surfaces')
        pattern_s = [geo.PlaneSurface(ll) for ll in pattern_ll]
        rve_s = geo.bool_cut_S(macro_s, pattern_s)
        rve_s = rve_s[0]
        logger.info('Done boolean operations on surfaces')
        rve_s_phy = geo.PhysicalGroup([rve_s], 2, "partition_plein")
        factory.synchronize()
        rve_s_phy.add_gmsh()
        factory.synchronize()

        data = model.getPhysicalGroups()
        logger.info('All physical groups in the model ' + repr(data)
                    + ' Names : ' + repr([model.getPhysicalName(*dimtag) for dimtag in data]))
        logger.info('Done generating the gmsh geometrical model')
        gmsh.write("%s.brep"%name)

        logger.info('Start defining a mesh refinement constraint')
        constr_pts = [pt for ll in pattern_ll for pt in ll.vertices]
        fine_pts = [pt for pt in constr_pts if (pt.coord[0] % 1 < p[0]/2. or pt.coord[0] % 1 > 1. - p[0]/2.)]
        fine_pts = geo.remove_duplicates(fine_pts)
        f = msh.set_mesh_refinement([r, a], [r/2, a/3], attractors={'points':fine_pts}, sigmoid_interpol=True)
        msh.set_background_mesh(f)
        logger.info('Done defining a mesh refinement constraint')
        
        macro_bndry = macro_ll.sides
        rve_s.get_boundary(recursive=True)
        micro_bndry = [geo.macro_line_fragments(rve_s.boundary, M_ln) for M_ln in macro_bndry]
        for i, crvs in enumerate(micro_bndry):
             msh.order_curves(crvs,
                macro_bndry[i%2].def_pts[-1].coord - macro_bndry[i%2].def_pts[0].coord,
                orientation=True)
        msh.set_periodicity_pairs(micro_bndry[0], micro_bndry[2])
        msh.set_periodicity_pairs(micro_bndry[1], micro_bndry[3])
        logger.info('Done defining a mesh periodicity constraint')
        geo.PhysicalGroup.set_group_mesh(True)
        model.mesh.generate(2)
        gmsh.write(f"{name}.msh")
        created_phy_gp = copy.deepcopy(geo.PhysicalGroup.all_groups)
        geo.reset()
        #modèle :  return Fenics_2D_RVE(GeneratingVectors, mesh ,listOfMaterials, domain = domain, symmetries = np.array((0,0)))
        return GeoAndMesh(gen_vect, created_phy_gp, name=name)