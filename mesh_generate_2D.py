# coding: utf-8
"""
Created on 17/01/2019
@author: Baptiste Durand, baptiste.durand@enpc.fr

Tools and classes designed to create 2D gmsh model of a periodic metamaterial represented on a certain domain.

The Gmsh2DRVE class has been specially designed for the generation of meshes that represent RVE with a given microstructure.
"""

import copy
import logging
import math
import os
from logging.handlers import RotatingFileHandler
from more_itertools import flatten, one

import numpy as np

import gmsh

import geometry as geo
import mesh_tools as msh

from subprocess import run

#TODO : placer un asarray dans la def de __init__ pour Point

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

def offset_pattern(cell_ll, offset, cell_vect):
    """
    Translation of the lineloops that define the microstructure geometry of a unit cell.

    Parameters
    ----------
    cell_ll : list of instances of LineLoop
    offset : 1D array
        relative coordinates with respect to the unit-cell generating vectors of the point that will be moved to the origin
    gen_vect : 2D array
        The generating vectors that are related to the given unit-cell.
    """
    if cell_vect.shape != (3,3):
        cell_vect_3D = np.zeros((3,3))
        cell_vect_3D[:cell_vect.shape[0],:cell_vect.shape[1]] = cell_vect
    else :
        cell_vect_3D = cell_vect
    offset_vect_relat = np.zeros(3)
    for i, val in enumerate(offset):
        offset_vect_relat[i] = val % 1.
    offset_vect_abs = np.dot(cell_vect_3D, offset_vect_relat)
    t_vect = -1 * offset_vect_abs
    shifted_ll = [geo.translation(ll, t_vect) for ll in cell_ll]
    return shifted_ll

class Gmsh2DRVE(object):
    #? Et si il y a pas seulement du mou et du vide mais plus de 2 matériaux constitutifs ? Imaginer une autre sous-classe semblable qui permet de définir plusieurs sous-domaines à partir d'une liste d'ensembles de LineLoop (chaque ensemble correspondant à un type d'inclusions ?)

    def __init__(self, pattern_ll, cell_vect, nb_cells, offset, attractors, soft_mat, name):
        """
        Contrat : Créer un maillage pour des RVE 2D, plans, comportant au plus 2 matériaux constitutifs et pouvant contenir plusieurs cellules.
        #! La cellule est un parallélogramme.

        Parameters
        ----------
        pattern_ll : list
            Instances of LineLoop that define the contours of the microstructure.
        cell_vect : 2D array
            dimensions of the unit cell and directions of periodicity.
            (given in a 2D cartesian coordinate system)
        nb_cells : 1D array
            Numbers of cells in each direction of repetition/periodicity.
        offset : 1D array
            Relative position inside a cell of the point that will coincide with the origin of the global domain.
        attractors : list
            Instances of Point.
            Can also be = None or empty.
            It represent the points that will be used as attractors in the definition of the element characteristic length fields.
            Attractors are geometrical elements of the cell around which mesh refinement constraints will be set.
            
        """
        self.name = name
        model.add(self.name)
        model.setCurrent(self.name)

        if offset.any():
            nb_pattern = [math.ceil(val+1) if offset[i] != 0 else math.ceil(val) for i,val in enumerate(nb_cells)]
            nb_pattern = np.array(nb_pattern, dtype=np.int8)
            pattern_ll = offset_pattern(pattern_ll, offset, cell_vect)
        else:
            nb_pattern = np.int8(np.ceil(nb_cells))
        
        if not np.equal(nb_pattern, 1).all():
                duplicate_pattern(pattern_ll, nb_pattern, cell_vect)
        
        rve_vect = cell_vect * nb_cells[:,np.newaxis]
        O = np.zeros((3,))
        macro_vtx = [O, rve_vect[0], rve_vect[0] + rve_vect[1] , rve_vect[1]]
        macro_ll = geo.LineLoop([geo.Point(c) for c in macro_vtx])
        macro_s = geo.PlaneSurface(macro_ll)

        if attractors:
            for entity in attractors:
                if not isinstance(entity, geo.Point):
                    raise TypeError(
                    """Use of curves as attractors for the refinement of the mesh
                    is not yet fully supported in our python library for gmsh.""")
            if offset.any():
                attractors = offset_pattern(attractors, offset, cell_vect)
            if not np.equal(nb_pattern, 1).all():
                duplicate_pattern(attractors, nb_pattern, cell_vect)

        logger.info('Start boolean operations on surfaces')
        phy_surf = list()
        pattern_s = [geo.PlaneSurface(ll) for ll in pattern_ll]
        rve_s = geo.AbstractSurface.bool_cut(macro_s, pattern_s)
        if len(rve_s) == 1:
            logger.info("The main material domain of the RVE is connected (topological property).")
        elif len(rve_s) == 0:
            logger.warning("The boolean operation for creating the main material domain of the RVE return 0 surfaces.")
        else:
            logger.warning("The main material domain of the RVE obtained by a boolean operation is disconnected (topological property).")
        rve_s_phy = geo.PhysicalGroup(rve_s, 2, "microstruct_domain")
        phy_surf.append(rve_s_phy)
        if soft_mat:
            # soft_s = geo.AbstractSurface.bool_intersect(macro_s, pattern_s) #! Abandon, cut rve_s plus efficace car évite la création d'une seconde frontière (éléments 1D)
            soft_s = geo.AbstractSurface.bool_cut(macro_s, rve_s)
            soft_s_phy = geo.PhysicalGroup(soft_s, 2, "soft_domain")
            phy_surf.append(soft_s_phy)
        logger.info('Done boolean operations on surfaces')

        if attractors:
            need_sync = False
            for entity in attractors:
                    if not entity.tag:
                        entity.add_gmsh()
                        need_sync = True
            if need_sync:
                factory.synchronize() #? Pourrait être enlevé ?
        
        for gp in phy_surf :
            gp.add_gmsh()
        factory.synchronize()

        data = model.getPhysicalGroups()
        details = [f"Physical group id : {dimtag[1]}, "
                   + f"dimension : {dimtag[0]}, "
                   + f"name : {model.getPhysicalName(*dimtag)}, "
                   + f"nb of entitities {len(model.getEntitiesForPhysicalGroup(*dimtag))} \n"
                   for dimtag in data]
        logger.debug(f"All physical groups in the model : {data}")
        logger.debug(f"Physical groups details : \n {details}")
        logger.info('Done generating the gmsh geometrical model')
        gmsh.write("%s.brep"%name)

        macro_bndry = macro_ll.sides
        if soft_mat:
            boundary = geo.AbstractSurface.get_surfs_boundary(rve_s+soft_s)
        else:
            try:
                s = one(rve_s)
                boundary = geo.AbstractSurface.get_surfs_boundary(s)
            except ValueError:
                boundary = geo.AbstractSurface.get_surfs_boundary(rve_s)
        factory.synchronize()
        micro_bndry = [geo.macro_line_fragments(boundary, M_ln) for M_ln in macro_bndry]
        macro_dir = [macro_bndry[i].def_pts[-1].coord - macro_bndry[i].def_pts[0].coord for i in range(len(macro_bndry)//2)]
        for i, crvs in enumerate(micro_bndry):
            msh.order_curves(crvs, macro_dir[i%2], orientation=True)
        msh.set_periodicity_pairs(micro_bndry[0], micro_bndry[2])
        msh.set_periodicity_pairs(micro_bndry[1], micro_bndry[3])
        logger.info('Done defining a mesh periodicity constraint')
        tags = ['per_pair_1_slave', 'per_pair_2_slave', 'per_pair_1_mast', 'per_pair_2_mast' ]
        per_pair_phy = list()
        for crvs, tag in zip(micro_bndry, tags):
            per_pair_phy.append(geo.PhysicalGroup(crvs, 1, tag))
        for gp in per_pair_phy:
            gp.add_gmsh()
        
        self.gen_vect = rve_vect
        self.nb_cells = nb_cells
        self.attractors = attractors if attractors else []
        self.phy_surf = phy_surf
        self.mesh_fields = []
        self.mesh_abs_path = ''

    def main_mesh_refinement(self, d_min_max, lc_min_max, sigmoid_interpol=False):
        model.setCurrent(self.name)
        attractors = {'points': self.attractors}
        logger.debug(f"When main_mesh_refinement(...) is called, physical groups in model : {model.getPhysicalGroups()}")
        rve_s = self.phy_surf[0].entities
        for s in rve_s:
            if not s.boundary:
                s.get_boundary()
        rve_boundary = list(flatten([s.boundary for s in rve_s]))
        restrict_domain = {
            'surfaces':rve_s,
            'curves':rve_boundary
            }
        field = msh.set_mesh_refinement(d_min_max, lc_min_max, attractors, 10,
                                        sigmoid_interpol, restrict_domain)
        self.mesh_fields.append(field)

    def soft_mesh_refinement(self, d_min_max, lc_min_max, sigmoid_interpol=False):
        model.setCurrent(self.name)
        attractors = {'points': self.attractors}
        soft_s = self.phy_surf[1].entities
        for s in soft_s:
            if not s.boundary:
                s.get_boundary()
        soft_boundary = list(flatten([s.boundary for s in soft_s]))
        restrict_domain = {
            'surfaces':soft_s,
            'curves':soft_boundary
            }
        field = msh.set_mesh_refinement(d_min_max, lc_min_max, attractors, 1,
                                        sigmoid_interpol, restrict_domain)
        self.mesh_fields.append(field)

    def mesh_generate(self, mesh_field=None):
        """Generate a 2D mesh of the model which represent a RVE.
        
        Parameters
        ----------
        mesh_field : mesh_tools.Field, optional
            The characteristic length of the elements can be explicitly prescribe by means of this field.
            The default is None. In this case, the fields that have been created with the soft_mesh_refinement and main_mesh_refinement methods are used.
        """
        
        model.setCurrent(self.name)
        if not mesh_field:
            self.background_field = msh.set_background_mesh(self.mesh_fields)
        else:
            self.background_field = msh.set_background_mesh(mesh_field)
        data = model.getPhysicalGroups()
        logger.debug(f'Physical groups in model just before generating mesh : {data}')
        geo.PhysicalGroup.set_group_mesh(True)
        model.mesh.generate(1)
        gmsh.model.mesh.removeDuplicateNodes()
        model.mesh.generate(2)
        gmsh.model.mesh.removeDuplicateNodes()
        geo.PhysicalGroup.set_group_visibility(False)
        gmsh.write(f"{self.name}.msh")
        self.mesh_abs_path = os.path.abspath(f"{self.name}.msh")

    @staticmethod
    def pantograph(a, b, k, junction_r, nb_cells=(1, 1), offset=(0., 0.), soft_mat=False, name=''):
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

        logger.info('Start defining the pantograph geometry')

        Lx = 4*a
        Ly = 6*a+2*b
        cell_vect = np.array(((Lx,0.), (0.,Ly)))

        e1 = np.array((a, 0., 0.))
        e2 = np.array((0., a, 0.))
        p = np.array((k, 0., 0.))
        b_ = b/a*e2
        E1 = geo.Point(e1)
        E2 = geo.Point(e2)
        E1m = geo.Point(-1*e1)
        E2m = geo.Point(-1*e2)
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
        sym_ll = [geo.plane_reflection(ll, I, e1) for ll in pattern_ll]
        for ll in sym_ll:
            ll.reverse()
        pattern_ll += sym_ll
        sym_ll = [geo.plane_reflection(ll, I, e2) for ll in pattern_ll]
        for ll in sym_ll:
            ll.reverse()
        pattern_ll += sym_ll
        pattern_ll = geo.remove_duplicates(pattern_ll)
        logger.info('Done removing of the line-loops duplicates')

        for ll in pattern_ll:
            ll.round_corner_incircle(junction_r)
        logger.info('Done rounding all corners of pattern line-loops')

        constr_pts = [pt for ll in pattern_ll for pt in ll.vertices]
        fine_pts = [pt for pt in constr_pts if (pt.coord[0] % 1 < p[0]/2. or pt.coord[0] % 1 > 1. - p[0]/2.)]
        fine_pts = geo.remove_duplicates(fine_pts)
        return Gmsh2DRVE(pattern_ll, cell_vect, nb_cells, offset, fine_pts, soft_mat, name)

    @staticmethod
    def auxetic_square(L, a, t, nb_cells=(1, 1), offset=(0., 0.), soft_mat=False, name=''):
        """
        Create an instance of Gmsh2DRVE that represents a RVE of the auxetic square microstructure.
        The geometry that is specific to this microstructure is defined in this staticmethod. Then, the generic operations will be performed with the Gmsh2DRVE methods.

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
            Relative position inside a cell of the point that will coincide with the origin of the global domain

        Returns
        -------
        Instance of the Gmsh2DRVE class.
        """

        name = name if name else "aux_square"
        model.add(name)
        # geo.reset()

        offset = np.asarray(offset)
        nb_cells = np.asarray(nb_cells)

        logger.info('Start defining the auxetic_square geometry')
        gen_vect = np.array(((L,0.), (0.,L)))
        b = (L - a)/2.
        e1 = np.array((L, 0., 0.))
        e2 = np.array((0., L, 0.))
        I = geo.Point(1/2.*(e1+e2))
        M = geo.Point(1/4.*(e1+e2))

        e3 = np.array((0., 0., 1.))

        center_pts = [[(b, 0.), (a+b, 0.)], [(0., -a/2.), (0., a/2.)]]
        center_pts = [[geo.Point(np.array(coord)) for coord in gp] for gp in center_pts]
        center_lines = [geo.Line(*pts) for pts in center_pts]
        center_lines += [geo.point_reflection(ln, M) for ln in center_lines]
        center_lines += [geo.plane_reflection(ln, I, e1) for ln in center_lines]
        center_lines += [geo.plane_reflection(ln, I, e2) for ln in center_lines]
        center_lines = geo.remove_duplicates(center_lines)

        for ln in center_lines:
            ln.ortho_dir = np.cross(e3, ln.direction())
        pattern_ll = list()
        for ln in center_lines:
            vertices = [
                geo.translation(ln.def_pts[0], t/2*ln.ortho_dir),
                geo.translation(ln.def_pts[1], t/2*ln.ortho_dir),
                geo.translation(ln.def_pts[1], -t/2*ln.ortho_dir),
                geo.translation(ln.def_pts[0], -t/2*ln.ortho_dir)
                ]
            pattern_ll.append(geo.LineLoop(vertices))
        tmp_nb_bef = len(pattern_ll)
        pattern_ll = geo.remove_duplicates(pattern_ll)

        logger.debug(f"Number of line-loops removed from pattern-ll : {tmp_nb_bef - len(pattern_ll)}.")
        logger.debug(f"Final number of pattern line-loops for auxetic square : {len(pattern_ll)}")

        for ll in pattern_ll:
            ll.round_corner_explicit(t/2)
            filter_sides = list() #* Pour ne pas essayer d'ajouter au model gmsh des lignes de longueur nulle. (Error : could not create line)
            for crv in ll.sides:
                if not crv.def_pts[0] == crv.def_pts[-1]:
                    filter_sides.append(crv)
            ll.sides = filter_sides

        fine_pts = geo.remove_duplicates(flatten([ln.def_pts for ln in center_lines]))

        return Gmsh2DRVE(pattern_ll, gen_vect, nb_cells, offset, fine_pts, soft_mat, name)

    @staticmethod
    def beam_pantograph(a, b, w, junction_r=0., nb_cells=(1, 1), offset=(0., 0.), soft_mat=False, name=''):
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

        logger.info('Start defining the beam pantograph geometry')
        Lx = 4*a
        Ly = 6*a+2*b
        cell_vect = np.array(((Lx,0.), (0.,Ly)))

        e1 = np.array((a, 0., 0.))
        e2 = np.array((0., a, 0.))
        b_ = b/a*e2
        E1 = geo.Point(e1)
        E2 = geo.Point(e2)
        E1m = geo.Point(-1*e1)
        E2m = geo.Point(-1*e2)
        L = geo.Point(2*(e1+e2))
        Lm = geo.Point(2*(e1-e2))
        M = geo.Point(e1 + 1.5*e2 + b_/2)
        I = geo.Point(2*(e1 + 1.5*e2 + b_/2))

        contours = [
            [E1, E2, E1m, E2m],
            [E1, Lm, geo.Point(3*e1), L],
            [E1, L, E2],
            [E2, L, geo.translation(L, b_), geo.translation(E2, b_)]
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