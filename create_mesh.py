# -*- coding: utf-8 -*-
"""
Created on 17/11/2018
@author: baptiste

"""

import logging
import math
import os
from logging.handlers import RotatingFileHandler

import matplotlib.pyplot as plt
import numpy as np
from more_itertools import one

import geometry as geo
import gmsh
import mesh_tools as msh

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
    if gen_vect.shape != (3,3):
        gen_vect_3D = np.zeros((3,3))
        gen_vect_3D[:gen_vect.shape[0],:gen_vect.shape[1]] = gen_vect
    else :
        gen_vect_3D = gen_vect
    # moins générique : gen_vect_3D = np.pad(a,((0, 1), (0, 1)),'constant',constant_values=0)
    # source : https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros

    for k in range(len(nb_cells)):
        if nb_cells[k] > 1 :
            new_contours = list()
            for i in range(1, int(nb_cells[k])):
                new_contours += [geo.translation(ll, i*gen_vect_3D[k]) for ll in repeated_ll]
            repeated_ll += new_contours
    repeated_ll = geo.remove_duplicates(repeated_ll)
    return repeated_ll

def offset_pattern(cell_ll, offset_vect, gen_vect):
    t_vect = -1 * np.array([val % gen_vect[i][i] for i,val in enumerate(offset_vect)])
        #GeneratingVector[i][i] pour récupérer la longueur de la cellule selon la direction i.
        #L'opératation modulo avec des floats est possible. Résultat du signe du second terme.
    shifted_ll = [geo.translation(ll, t_vect) for ll in cell_ll]
    return shifted_ll

# class GeoAndMesh_1(object): #! On crée plutôt des objets Part maillage+matériau
#     """
#     Contrat :
#     Pour un certain choix de microstructure, construit un mesh qui représente un milieu périodique sur une certaine étendue.

#     Une instance contient :
#     name : string
#         le nom du maillage ou le chemin complet vers le maillage
#     physical_groups : dictionnary
#         a dictionnary that catalogs all the physical groups that have been created and exported in the msh file.
#     gen_vect : square array, 2D or 3D
#         The generating vectors that are related to the given microstruture.

#     """
#     def __init__(self,):
#         self.mesh_path = #? ou name ?
#         self.physical_groups =
#     # def __init__(self, cell_ll): #? En entrée : lineloop qui composent exactement le motif(donc après arrondi ou offset si nécessaire) sur une ce


#     @staticmethod
#     def pantograph(cell_dim, junction_r, nb_cells=(1, 1), offset=(0., 0.), soft_mat=False, name=''):
#         """
#         Contrat :
#         Pour un certain choix de microstructure, construit un mesh qui représente un milieu périodique sur une certaine étendue. 
#         Parameters
#         ----------
#         cell_dim : dictionnary
#             main length of the microstruture
#         junction_r : float
#             radius of the junctions inside the pantograph microstructure
#         nb_cells : tuple
#             nb of cells in each direction of repetition
#         offset : tuple or array
#             Relative position inside a cell of the point that will coincide with the origin of the global domain
#         Returns
#         -------
#         Instance of the GeoAndMesh class.
#         """

#         name = name if name else "pantograph" 
#         model.add(name)
#         geo.reset()

#         logger.info('Start defining the pantograph geometry')
#         a = cell_dim['a']
#         b = cell_dim['b']
#         k = cell_dim['k']
#         r = junction_r

#         Lx = 4*a
#         Ly = 6*a+2*b
#         gen_vect = np.array(((Lx,0.), (0.,Ly)))

#         e1 = np.array((a, 0., 0.))
#         e2 = np.array((0., a, 0.))
#         p = np.array((k, 0., 0.))
#         b = b/a*e2
#         E1 = geo.Point(e1)
#         E2 = geo.Point(e2)
#         E1m = geo.Point(-1*e1)
#         E2m = geo.Point(-1*e2)
#         O = np.zeros((3,))
#         L = geo.Point(2*(e1+e2))
#         Lm = geo.Point(2*(e1-e2))
#         M = geo.Point(e1 + 1.5*e2 + b/2)
#         I = geo.Point(2*(e1 + 1.5*e2 + b/2))

#         contours = list()
#         contours.append([E1, E2, E1m, E2m])
#         contours.append([E1, Lm, geo.Point(3*e1), L])
#         contours.append([E2, L, geo.translation(L,b/2-p), geo.translation(L,b), geo.translation(E2,b), geo.translation(E2,b/2+p)])
#         pattern_ll = [geo.LineLoop(pt_list, explicit=False) for pt_list in contours]

#         #? Créer une méthode statique pour réaliser ces opérations de symétries. Parameters : la liste de lineloop et une liste qui indique les opérations de symétries
#         pattern_ll += [geo.point_reflection(ll, M) for ll in pattern_ll]
#         pattern_ll += [geo.plane_reflection(ll, I, e1) for ll in pattern_ll]
#         pattern_ll += [geo.plane_reflection(ll, I, e2) for ll in pattern_ll]
#         geo.remove_duplicates(pattern_ll)
#         logger.info('Done removing of the line-loops duplicates')
#         #! REPRENDRE ICI
#         #? Idem, utiliser un array pour offset ?
#         if np.asarray(offset).any():
#             effect_nb_cells = [int(math.ceil(val+1)) if offset_v[i] != 0 else int(math.ceil(val)) for i,val in enumerate(nb_cells)]
#             pattern_ll = offset_cell(pattern_ll, offset_v, gener_vect)
#         else:
#             nb_cells_ = [int(math.ceil(val)) for val in nb_cells]
   
#         #? Utiliser un array pour nb_cells ?
#         if not np.equal(np.asarray(nb_cells),1).all():
#             duplicate_pattern(pattern_ll, nb_cells_, gen_vect)
        
#         for ll in pattern_ll:
#             ll.round_corner_incircle(r)
#         logger.info('Done rounding all corners of pattern line-loops')

#         macro_vtcs = [O, gen_vect[0], gen_vect[0] + gen_vect[1] , gen_vect[1]] #TODO là il faut modifier pour prendre en compte le nb de cellules. Ce n'est pas gen_vect mais un multiple.
#         macro_ll = geo.LineLoop([geo.Point(c) for c in macro_vtcs])
#         macro_s = geo.PlaneSurface(macro_ll)

#         logger.info('Start boolean operations on surfaces')
#         pattern_s = [geo.PlaneSurface(ll) for ll in pattern_ll]
#         rve_s = geo.bool_cut_S(macro_s, pattern_s)
#         rve_s = rve_s[0]
#         logger.info('Done boolean operations on surfaces')
#         rve_s_phy = geo.PhysicalGroup([rve_s], 2, "partition_plein")
#         factory.synchronize()
#         rve_s_phy.add_gmsh()
#         factory.synchronize()

#         data = model.getPhysicalGroups()
#         logger.info('All physical groups in the model ' + repr(data)
#                     + ' Names : ' + repr([model.getPhysicalName(*dimtag) for dimtag in data]))
#         logger.info('Done generating the gmsh geometrical model')
#         gmsh.write("%s.brep"%name)

#         logger.info('Start defining a mesh refinement constraint')
#         constr_pts = [pt for ll in pattern_ll for pt in ll.vertices]
#         fine_pts = [pt for pt in constr_pts if (pt.coord[0] % 1 < p[0]/2. or pt.coord[0] % 1 > 1. - p[0]/2.)]
#         fine_pts = geo.remove_duplicates(fine_pts)
#         f = msh.set_mesh_refinement([r, a], [r/2, a/3], attractors={'points':fine_pts}, sigmoid_interpol=True)
#         msh.set_background_mesh(f)
#         logger.info('Done defining a mesh refinement constraint')
        
#         macro_bndry = macro_ll.sides
#         rve_s.get_boundary(recursive=True)
#         micro_bndry = [geo.macro_line_fragments(rve_s.boundary, M_ln) for M_ln in macro_bndry]
#         for i, crvs in enumerate(micro_bndry):
#              msh.order_curves(crvs,
#                 macro_bndry[i%2].def_pts[-1].coord - macro_bndry[i%2].def_pts[0].coord,
#                 orientation=True)
#         msh.set_periodicity_pairs(micro_bndry[0], micro_bndry[2])
#         msh.set_periodicity_pairs(micro_bndry[1], micro_bndry[3])
#         logger.info('Done defining a mesh periodicity constraint')
#         geo.PhysicalGroup.set_group_mesh(True)
#         model.mesh.generate(2)
#         gmsh.write(f"{name}.msh")
#         created_phy_gp = copy.deepcopy(geo.PhysicalGroup.all_groups)
#         geo.reset()
#         #modèle :  return Fenics_2D_RVE(GeneratingVectors, mesh ,listOfMaterials, domain = domain, symmetries = np.array((0,0)))
#         return GeoAndMesh1(gen_vect, created_phy_gp, name=name)


class FenicsPart(object):
    pass

class Fenics2DRVE(object): #? Et si il y a pas seulement du mou et du vide mais plus de 2 matériaux constitutifs ? Imaginer une autre sous-classe semblable qui permet de définir plusieurs sous-domaines à partir d'une liste d'ensembles de LineLoop (chaque ensemble correspondant à un type d'inclusions ?)
    
    def __init__(self, pattern_ll, gen_vect, nb_cells, offset, attractors, soft_mat, name):
        """
        Contrat : Créer un couple maillage + matériaux pour des RVE 2D, plans, comportant au plus 2 matériaux constitutifs et pouvant contenir plusieurs cellules.
        #! La cellule est un parallélogramme.

        #! Pour le moment, seule la géométrie est crée.

        Parameters
        ----------
        pattern_ll : list
            Instances of LineLoop that define the contours of the microstructure.
        gen_vect : 2D array
            dimensions of the unit cell and directions of periodicity.
            (given in a 2D cartesian coordinate system)
        nb_cells : 1D array
            Numbers of cells in each direction of repetition/periodicity.
        offset : 1D array
            Relative position inside a cell of the point that will coincide with the origin of the global domain.
        # refinement : instance of a Field subclass
        #     Scalar field that defines an element size constraint for the mesh generation.
        # attractors : instances of Point or Curve #? Ou dict {'points':[],'curves':[]} ? # ou list of physical groups ?
            Geometrical elements of the cell around which mesh refinement constraints will be set.
        """

        if offset.any():
            nb_pattern = [math.ceil(val+1) if offset[i] != 0 else math.ceil(val) for i,val in enumerate(nb_cells)]
            nb_pattern = np.array(nb_pattern, dtype=np.int8) #? Par confort de travailler avec un array ensuite (test np.equal). Est-ce gênant ?
            pattern_ll = offset_pattern(pattern_ll, offset, gen_vect)
        else:
            nb_pattern = np.int8(np.ceil(nb_cells))
        if not np.equal(nb_pattern, 1).all():
                duplicate_pattern(pattern_ll, nb_pattern, gen_vect)
        macro_vect = gen_vect*nb_cells[:,np.newaxis]
        O = np.zeros((3,))
        macro_vtx = [O, macro_vect[0], macro_vect[0] + macro_vect[1] , macro_vect[1]]
        macro_ll = geo.LineLoop([geo.Point(c) for c in macro_vtx])
        macro_s = geo.PlaneSurface(macro_ll)
        
        for attract_gp in attractors:
            if offset.any():
                attract_gp.entities = offset_pattern(attract_gp.entities, offset, gen_vect)
            if not np.equal(nb_pattern, 1).all():
                duplicate_pattern(attract_gp.entities, nb_pattern, gen_vect)

        logger.info('Start boolean operations on surfaces')
        phy_surf = list()
        pattern_s = [geo.PlaneSurface(ll) for ll in pattern_ll]
        rve_s = geo.bool_cut_S(macro_s, pattern_s)
        rve_s = rve_s[0]
        rve_s_phy = geo.PhysicalGroup([rve_s], 2, "microstruct_domain")
        phy_surf.append(rve_s_phy)
        if soft_mat:
            soft_s = geo.bool_intersect_S(macro_s, pattern_s)
            soft_s = soft_s[0]
            soft_s_phy = geo.PhysicalGroup([rve_s], 2, "soft_domain")
            phy_surf.append(soft_s_phy)
        logger.info('Done boolean operations on surfaces')
        factory.synchronize()
        rve_s_phy.add_gmsh()
        if soft_mat:
            soft_s_phy.add_gmsh()
        for attract_gp in attractors:
            attract_gp.add_gmsh()
        factory.synchronize()

        data = model.getPhysicalGroups()
        logger.info('All physical groups in the model ' + repr(data)
                    + ' Names : ' + repr([model.getPhysicalName(*dimtag) for dimtag in data]))
        logger.info('Done generating the gmsh geometrical model')
        gmsh.write("%s.brep"%name)

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

        self.name = name
        self.gen_vect = gen_vect
        self.nb_cells = nb_cells
        self.macro_vect = macro_vect
        self.attractors = attractors
        self.phy_surf = phy_surf

    def main_mesh_refinement(self, d_min_max, lc_min_max, nb_pts_discretization=10, sigmoid_interpol=False):
        model.setCurrent(self.name)
        attractors = {'points':[],'curves':[]}
        for attract_gp in self.attractors:
            if attract_gp.dim == 0:
                attractors['points'] += attract_gp.entities
            elif attract_gp.dim == 1:
                attractors['curves'] += attract_gp.entities
        rve_s = one(self.phy_surf[0].entities)
        restrict_domain = {'surfaces':[rve_s]}
        if not rve_s.boundary:
            factory.synchronize()
            rve_s.get_boundary(recursive=False)
        restrict_domain['curves'] = rve_s.boundary #? Duck typing, même les PlaneSurface ont un attribut boundary
        field = msh.set_mesh_refinement(d_min_max, lc_min_max, attractors, nb_pts_discretization, sigmoid_interpol, restrict_domain)
        try:
            self.mesh_fields.append(field)
        except AttributeError:
            self.mesh_fields = [field]
    
    def soft_mesh_refinement(self, d_min_max, lc_min_max, nb_pts_discretization=10, sigmoid_interpol=False):
        model.setCurrent(self.name)
        attractors = {'points':[],'curves':[]}
        for attract_gp in self.attractors:
            if attract_gp.dim == 0:
                attractors['points'] += attract_gp.entities
            elif attract_gp.dim == 1:
                attractors['curves'] += attract_gp.entities
        soft_s = one(self.phy_surf[1].entities)
        restrict_domain = {'surfaces':[soft_s]}
        field = msh.set_mesh_refinement(d_min_max, lc_min_max, attractors, nb_pts_discretization, sigmoid_interpol, restrict_domain)
        try:
            self.mesh_fields.append(field)
        except AttributeError:
            self.mesh_fields = [field]

    def mesh_generate(self):
        model.setCurrent(self.name)
        self.mesh_fields = msh.set_background_mesh(self.mesh_fields)
        geo.PhysicalGroup.set_group_mesh(True)
        model.mesh.generate(2)
        gmsh.write(f"{self.name}.msh")
        #TODO : import mesh in a fenics object

    @staticmethod
    def pantograph(a,b,k, junction_r, nb_cells=(1, 1), offset=(0., 0.), soft_mat=False, name=''):
        """
        Contrat :
        
        Parameters
        ----------
        a,b and k : floats
            main length of the microstruture
        junction_r : float
            radius of the junctions inside the pantograph microstructure
        nb_cells : tuple or 1D array
            nb of cells in each direction of repetition
        offset : tuple or 1D array
            Relative position inside a cell of the point that will coincide with the origin of the global domain
        #! Et les paramètres de raffinement du maillage ?

        Returns
        -------
        Instance of the Fenics2DRVE class.
        """

        name = name if name else "pantograph" 
        model.add(name)
        geo.reset()

        offset = np.asarray(offset)
        nb_cells = np.asarray(nb_cells)

        logger.info('Start defining the pantograph geometry')

        Lx = 4*a
        Ly = 6*a+2*b
        gen_vect = np.array(((Lx,0.), (0.,Ly)))

        e1 = np.array((a, 0., 0.))
        e2 = np.array((0., a, 0.))
        p = np.array((k, 0., 0.))
        b_ = b/a*e2
        E1 = geo.Point(e1)
        E2 = geo.Point(e2)
        E1m = geo.Point(-1*e1)
        E2m = geo.Point(-1*e2)
        O = np.zeros((3,))
        L = geo.Point(2*(e1+e2))
        Lm = geo.Point(2*(e1-e2))
        M = geo.Point(e1 + 1.5*e2 + b_/2)
        I = geo.Point(2*(e1 + 1.5*e2 + b_/2))

        contours = list()
        contours.append([E1, E2, E1m, E2m])
        contours.append([E1, Lm, geo.Point(3*e1), L])
        contours.append([E2, L, geo.translation(L, 0.5*b_-p), geo.translation(L, b_), geo.translation(E2, b_), geo.translation(E2, 0.5*b_+p)])
        pattern_ll = [geo.LineLoop(pt_list, explicit=False) for pt_list in contours]

        pattern_ll += [geo.point_reflection(ll, M) for ll in pattern_ll]
        pattern_ll += [geo.plane_reflection(ll, I, e1) for ll in pattern_ll]
        pattern_ll += [geo.plane_reflection(ll, I, e2) for ll in pattern_ll]
        pattern_ll = geo.remove_duplicates(pattern_ll)
        logger.info('Done removing of the line-loops duplicates')

        for ll in pattern_ll:
            ll.round_corner_incircle(junction_r)
        logger.info('Done rounding all corners of pattern line-loops')

        constr_pts = [pt for ll in pattern_ll for pt in ll.vertices]
        fine_pts = [pt for pt in constr_pts if (pt.coord[0] % 1 < p[0]/2. or pt.coord[0] % 1 > 1. - p[0]/2.)]
        fine_pts = geo.remove_duplicates(fine_pts)
        mesh_attract = geo.PhysicalGroup(fine_pts,0,'mesh_attractors')
        attractors= [mesh_attract]
        return Fenics2DRVE(pattern_ll, gen_vect, nb_cells, offset, attractors, soft_mat, name)

    @staticmethod
    def auxetic_square(a, L, t, junction_r, nb_cells=(1, 1), offset=(0., 0.), soft_mat=False, name=''):
        """
        Contrat :
        
        Parameters
        ----------
        L : float
            Length of the sides of the square cell.
        a : float
            Length of the slits beetween squares.
        t : float
            Size of ????#TODO : à déterminer
        nb_cells : tuple or 1D array
            nb of cells in each direction of repetition
        offset : tuple or 1D array
            Relative position inside a cell of the point that will coincide with the origin of the global domain

        Returns
        -------
        Instance of the Fenics2DRVE class.
        """

        name = name if name else "aux_square" 
        model.add(name)
        geo.reset()

        offset = np.asarray(offset)
        nb_cells = np.asarray(nb_cells)

        logger.info('Start defining the pantograph geometry')
        gen_vect = np.array(((L,0.), (0.,L)))
        b = (L - a)/2.
        e1 = np.array((L, 0., 0.))
        e2 = np.array((0., L, 0.))
        I = geo.Point(1/2.*(e1+e2))
        M = geo.Point(1/4.*(e1+e2))
        
        ends = [[(b, -t/2.), (a+b, t/2.)], [(-t/2., -a/2.), (t/2., a/2.)]]
        contours = list()
        for pt_l in ends:
            vtcs = [pt_l[0], (pt_l[1][0], pt_l[0][1]), pt_l[1], (pt_l[0][0], pt_l[1][1])]
            contours.append([geo.Point(np.array(c)) for c in vtcs])
        pattern_ll = [geo.LineLoop(pt_list, explicit=False) for pt_list in contours]
        pattern_ll += [geo.point_reflection(ll, M) for ll in pattern_ll]
        pattern_ll += [geo.plane_reflection(ll, I, e1) for ll in pattern_ll]
        pattern_ll += [geo.plane_reflection(ll, I, e2) for ll in pattern_ll]
        geo.remove_duplicates(pattern_ll)
        logger.info('Done removing of the line-loops duplicates')

        attract_lines = [geo.Line(geo.Point(np.array(pts[0])), geo.Point(np.array(pts[1]))) for pts in ends]
        mesh_attract = geo.PhysicalGroup(attract_lines,1,'mesh_attractors')
        attractors = [mesh_attract]

        return Fenics2DRVE(pattern_ll, gen_vect, nb_cells, offset, attractors, soft_mat, name)

if __name__ == "__main__":
    geo.init_geo_tools()
    panto_test = Fenics2DRVE.pantograph(1,1,0.3,0.1,(2,3),False,'panto_test')
    panto_test.main_mesh_refinement((0.1,0.5),(0.03,0.3),False)
    panto_test.mesh_generate()
    os.system(f"gmsh {panto_test.name}.msh &")

    t = 0.05
    L = 1
    a = L-3*t
    aux_sqr_test = Fenics2DRVE.auxetic_square(a, L, t, 0.2)