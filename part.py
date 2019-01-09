# -*- coding: utf-8 -*-
"""
Created on 17/11/2018
@author: baptiste

"""

import copy
import logging
import math
import os
from logging.handlers import RotatingFileHandler
from more_itertools import flatten, one

import matplotlib.pyplot as plt
import numpy as np

import dolfin as fe
import gmsh

import geometry as geo
import materials as mat
import mesh_tools as msh

from subprocess import run

#TODO : placer un asarray dans la def de __init__ pour Point

# nice shortcuts
model = gmsh.model
factory = model.occ

plt.ioff()

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


def facet_plot2d(facet_func,mesh, mesh_edges=True, markers=None, exclude_val=(0,), **kargs):
    """
    Source : https://bitbucket.org/fenics-project/dolfin/issues/951/plotting-facetfunctions-in-matplotlib-not
    """
    x_list, y_list = [],[]
    if markers == None:
        for facet in fe.facets(mesh):
            mp = facet.midpoint()
            x_list.append(mp.x())
            y_list.append(mp.y())
        values = facet_func.array()
    else:
        i = 0
        values = []
        for facet in fe.facets(mesh):
            if facet_func[i] in markers:
                mp = facet.midpoint()
                x_list.append(mp.x())
                y_list.append(mp.y())
                values.append(facet_func[i])
            i+=1
    if exclude_val:
        filtered_data = [], [], []
        for x, y, val in zip(x_list, y_list, values):
            if val in exclude_val:
                continue
            filtered_data[0].append(x)
            filtered_data[1].append(y)
            filtered_data[2].append(val)
        x_list, y_list, values = filtered_data

    plots = [plt.scatter(x_list, y_list, s=30, c=values, linewidths=1, **kargs)]
    if mesh_edges:
        plots.append(fe.plot(facet_func.mesh()))
    return plots


class FenicsPart(object):
    pass

class Fenics2DRVE(FenicsPart):
    """
    Contrat : Créer un couple maillage + matériaux pour des RVE 2D, plans, comportant au plus 2 matériaux constitutifs et pouvant contenir plusieurs cellules.
    """
    def __init__(self, mesh, ):
        self.mesh = mesh


    @staticmethod
    def gmsh_2_Fenics_2DRVE(gmsh_2D_RVE):
        """
        Generate an instance of Fenics2DRVE from a instance of the Gmsh2DRVE class.

        """
        print(os.getcwd())
        mesh = fe.Mesh()
        os.system(f'dolfin-convert {gmsh_2D_RVE.mesh_abs_path} {gmsh_2D_RVE.name}.xml')
        mesh = fe.Mesh(f'{gmsh_2D_RVE.name}.xml')
        plt.figure()
        fe.plot(mesh, "2D mesh")
        plt.show()
        return Fenics2DRVE(mesh)

class Gmsh2DRVE(object): #? Et si il y a pas seulement du mou et du vide mais plus de 2 matériaux constitutifs ? Imaginer une autre sous-classe semblable qui permet de définir plusieurs sous-domaines à partir d'une liste d'ensembles de LineLoop (chaque ensemble correspondant à un type d'inclusions ?)

    def __init__(self, pattern_ll, cell_vect, nb_cells, offset, attractors, soft_mat, name):
        """
        Contrat : Créer un couple maillage + matériaux pour des RVE 2D, plans, comportant au plus 2 matériaux constitutifs et pouvant contenir plusieurs cellules.
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
        # refinement : instance of a Field subclass
        #     Scalar field that defines an element size constraint for the mesh generation.
        attractors : list of instances of PhysicalGroups
            Physical groups that represent groups of points that will be used as attractors in the definition of the element characteristic length field. #? Ou dict {'points':[],'curves':[]} ? # ou list of physical groups ?
            Geometrical elements of the cell around which mesh refinement constraints will be set.
        """
        self.name = name
        model.add(self.name)
        model.setCurrent(self.name)

        if offset.any():
            nb_pattern = [math.ceil(val+1) if offset[i] != 0 else math.ceil(val) for i,val in enumerate(nb_cells)]
            nb_pattern = np.array(nb_pattern, dtype=np.int8) #? Par confort de travailler avec un array ensuite (test np.equal). Est-ce gênant ?
            pattern_ll = offset_pattern(pattern_ll, offset, cell_vect)
        else:
            nb_pattern = np.int8(np.ceil(nb_cells))
        if not np.equal(nb_pattern, 1).all():
                duplicate_pattern(pattern_ll, nb_pattern, cell_vect)
        rve_vect = cell_vect*nb_cells[:,np.newaxis]
        O = np.zeros((3,))
        macro_vtx = [O, rve_vect[0], rve_vect[0] + rve_vect[1] , rve_vect[1]]
        macro_ll = geo.LineLoop([geo.Point(c) for c in macro_vtx])
        macro_s = geo.PlaneSurface(macro_ll)

        for attract_gp in attractors:
            if attract_gp.dim != 0:
                raise TypeError(
                """Use of curves as attractors for the refinement of the mesh
                is not yet fully supported in our python library for gmsh.""")
        for attract_gp in attractors:
            if offset.any():
                attract_gp.entities = offset_pattern(attract_gp.entities, offset, cell_vect)
            if not np.equal(nb_pattern, 1).all():
                duplicate_pattern(attract_gp.entities, nb_pattern, cell_vect)

        logger.info('Start boolean operations on surfaces')
        phy_surf = list()
        pattern_s = [geo.PlaneSurface(ll) for ll in pattern_ll]
        rve_s = geo.AbstractSurface.bool_cut(macro_s, pattern_s)
        rve_s = rve_s[0]
        rve_s_phy = geo.PhysicalGroup([rve_s], 2, "microstruct_domain")
        phy_surf.append(rve_s_phy)
        if soft_mat:
            soft_s = geo.AbstractSurface.bool_intersect(macro_s, pattern_s)
            soft_s = soft_s[0]
            soft_s_phy = geo.PhysicalGroup([rve_s], 2, "soft_domain")
            phy_surf.append(soft_s_phy)
        logger.info('Done boolean operations on surfaces')
        factory.synchronize()

        rve_s.get_boundary(recursive=True)
        rve_bound_phy = geo.PhysicalGroup(rve_s.boundary, 1, "main_domain_bound")
        factory.synchronize()

        need_sync = False
        for attract_gp in attractors:
            for ent in attract_gp.entities:
                if not ent.tag:
                    ent.add_gmsh()
                    need_sync = True
        if need_sync:
            factory.synchronize()
        for gp in attractors + phy_surf + [rve_bound_phy]:
            gp.add_gmsh()
        factory.synchronize()

        data = model.getPhysicalGroups()
        details = [f"Physical group id : {dimtag[1]}, "
                   + f"dimension : {dimtag[0]}, "
                   + f"name : {model.getPhysicalName(*dimtag)}, "
                   + f"nb of entitities {len(model.getEntitiesForPhysicalGroup(*dimtag))} \n"
                   for dimtag in data]
        logger.info(f"All physical groups in the model : {data}")
        logger.info(f"Physical groups details : \n {details}")
        logger.info('Done generating the gmsh geometrical model')
        gmsh.write("%s.brep"%name)

        macro_bndry = macro_ll.sides
        micro_bndry = [geo.macro_line_fragments(rve_s.boundary, M_ln) for M_ln in macro_bndry]
        macro_dir = [macro_bndry[i].def_pts[-1].coord - macro_bndry[i].def_pts[0].coord for i in range(len(macro_bndry)//2)]
        for i, crvs in enumerate(micro_bndry):
            msh.order_curves(crvs, macro_dir[i%2], orientation=True)
        msh.set_periodicity_pairs(micro_bndry[0], micro_bndry[2])
        msh.set_periodicity_pairs(micro_bndry[1], micro_bndry[3])
        logger.info('Done defining a mesh periodicity constraint')

        self.gen_vect = rve_vect
        self.nb_cells = nb_cells
        self.attractors = attractors
        self.phy_surf = phy_surf
        self.mesh_abs_path = ''

    def main_mesh_refinement(self, d_min_max, lc_min_max, sigmoid_interpol=False):
        model.setCurrent(self.name)
        attractors = {'points':[]}
        logger.debug(f"When main_mesh_refinement(...) is called, physical groups in model : {model.getPhysicalGroups()}")
        for attract_gp in self.attractors:
            # logger.debug(f"Physical group of attractors used in main_mesh_refinement : {attract_gp.__dict__}")
            if attract_gp.dim == 0:
                attractors['points'] += attract_gp.entities
            else :
                raise TypeError('Only points can be used as attractors.')
        rve_s = one(self.phy_surf[0].entities)
        restrict_domain = {
            'surfaces':[rve_s],
            'curves':rve_s.boundary
            }
        field = msh.set_mesh_refinement(d_min_max, lc_min_max, attractors, 1,
                                        sigmoid_interpol, restrict_domain)
        try:
            self.mesh_fields.append(field)
        except AttributeError:
            self.mesh_fields = [field]

    def soft_mesh_refinement(self, d_min_max, lc_min_max, sigmoid_interpol=False):
        model.setCurrent(self.name)
        attractors = {'points':[]}
        for attract_gp in self.attractors:
            # logger.debug(f"Physical group of attractors used in soft_mesh_refinement : {attract_gp.__dict__}")
            if attract_gp.dim == 0:
                attractors['points'] += attract_gp.entities
            else :
                raise TypeError('Only points can be used as attractors.')
        soft_s = one(self.phy_surf[1].entities)
        restrict_domain = {'surfaces':[soft_s]}
        field = msh.set_mesh_refinement(d_min_max, lc_min_max, attractors, 1,
                                        sigmoid_interpol, restrict_domain)
        try:
            self.mesh_fields.append(field)
        except AttributeError:
            self.mesh_fields = [field]

    def mesh_generate(self):
        model.setCurrent(self.name)
        self.mesh_fields = msh.set_background_mesh(self.mesh_fields)
        data = model.getPhysicalGroups()
        logger.info(f'Physical groups in model just before generating mesh : {data}')
        geo.PhysicalGroup.set_group_mesh(True)
        model.mesh.generate(1)
        model.mesh.generate(2)
        geo.PhysicalGroup.set_group_visibility(False)
        logger.debug(f"value of Mesh.SaveAll option before writting {self.name}.msh : {gmsh.option.getNumber('Mesh.SaveAll')}")
        gmsh.write(f"{self.name}.msh")
        self.mesh_abs_path = os.path.abspath(f"{self.name}.msh")

    @staticmethod
    def pantograph(a, b, k, junction_r, nb_cells=(1, 1), offset=(0., 0.), soft_mat=False, name=''):
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

        Returns
        -------
        Instance of the Gmsh2DRVE class.
        """

        name = name if name else "pantograph"
        # geo.reset()

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
        mesh_attract = geo.PhysicalGroup(fine_pts, 0, 'mesh_attractors')
        attractors= [mesh_attract]

        return Gmsh2DRVE(pattern_ll, cell_vect, nb_cells, offset, attractors, soft_mat, name)

    @staticmethod
    def auxetic_square(a, L, t, nb_cells=(1, 1), offset=(0., 0.), soft_mat=False, name=''):
        """
        Contrat :

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
        tmp_nb_aft = len(pattern_ll)
        logger.info(f"Number of line-loops removed from pattern-ll : {tmp_nb_bef - tmp_nb_aft}."
            +f"Final number of pattern line-loops for auxetic square : {tmp_nb_aft}")
        for ll in pattern_ll:
            ll.round_corner_explicit(t/2)
            filter_sides = list() #* Pour ne pas essayer d'ajouter au model gmsh des lignes de longueur nulle. (Error : could not create line)
            for crv in ll.sides:
                if not crv.def_pts[0] == crv.def_pts[-1]:
                    filter_sides.append(crv)
            ll.sides = filter_sides

        fine_pts = geo.remove_duplicates(flatten([ln.def_pts for ln in center_lines]))
        mesh_attract = geo.PhysicalGroup(fine_pts, 0, 'mesh_attractors')
        attractors = [mesh_attract]

        return Gmsh2DRVE(pattern_ll, gen_vect, nb_cells, offset, attractors, soft_mat, name)

    @staticmethod
    def beam_pantograph(a, b, w, junction_r=0, nb_cells=(1, 1), offset=(0., 0.), soft_mat=False, name=''):
        """
        junction_r : float, optional
        if = 0 or False, the angles will not be rounded.

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
        mesh_attract = geo.PhysicalGroup(fine_pts, 0, 'mesh_attractors')
        attractors= [mesh_attract]

        return Gmsh2DRVE(pattern_ll, cell_vect, nb_cells, offset, attractors, soft_mat, name)


if __name__ == "__main__":
    geo.init_geo_tools()

    # a = 1
    # b, k = a, a/3
    # panto_test = Gmsh2DRVE.pantograph(a, b, k, 0.1, nb_cells=(2, 3), soft_mat=False, name='panto_test')
    # panto_test.main_mesh_refinement((0.1,0.5),(0.03,0.3),False)
    # panto_test.mesh_generate()
    # os.system(f"gmsh {panto_test.name}.msh &")

    # a = 1
    # b, k = a, a/3
    # panto_test_offset = Gmsh2DRVE.pantograph(a, b, k, 0.1, nb_cells=(2,3), offset=(0.25,0.25), soft_mat=False, name='panto_test_offset')
    # panto_test_offset.main_mesh_refinement((0.1,0.5),(0.03,0.3),False)
    # panto_test_offset.mesh_generate()
    # os.system(f"gmsh {panto_test_offset.name}.msh &")


    # L, t = 1, 0.05
    # a = L-3*t
    # aux_sqr_test = Gmsh2DRVE.auxetic_square(a, L, t, nb_cells=(4,3), soft_mat=False, name='aux_square_test')
    # os.system(f"gmsh {aux_sqr_test.name}.brep &")
    # aux_sqr_test.main_mesh_refinement((0.1,0.3), (0.01,0.05), False)
    # aux_sqr_test.mesh_generate()
    # os.system(f"gmsh {aux_sqr_test.name}.msh &")

    # a = 1
    # b = a
    # w = a/50
    # r = 4*w
    # beam_panto_test = Gmsh2DRVE.beam_pantograph(a, b, w, r, nb_cells=(1, 1), offset=(0., 0.), soft_mat=False, name='beam_panto_test')
    # os.system(f"gmsh {beam_panto_test.name}.brep &")
    # beam_panto_test.main_mesh_refinement((5*w, a/2),(w/5, w), True)
    # beam_panto_test.mesh_generate()
    # os.system(f"gmsh {beam_panto_test.name}.msh &")

    a = 1
    b, k = a, a/3
    panto_test = Gmsh2DRVE.pantograph(a, b, k, 0.1, nb_cells=(1, 1), soft_mat=False, name='panto_test_2')
    panto_test.main_mesh_refinement((0.1,0.5),(0.03,0.3),False)
    panto_test.mesh_generate()
    try :
        os.system(f"gmsh {panto_test.name}.msh &")
        Fenics2DRVE.gmsh_2_Fenics_2DRVE(panto_test)
        logging.critical('Conversion and import as a Mesh() object done.')
        logging.info(f"Mesh.SaveAll value : {gmsh.option.getNumber('Mesh.SaveAll')}")
    except RuntimeError:
        logging.critical('Conversion and import as a Mesh() object has failed.')
        logging.info(f"Mesh.SaveAll value : {gmsh.option.getNumber('Mesh.SaveAll')}")

    gmsh.option.setNumber('Mesh.SurfaceFaces',1) #Display faces of surface mesh?
    gmsh.fltk.run()

# msh.set_background_mesh(field)

#         gmsh.option.setNumber('Mesh.CharacteristicLengthExtendFromBoundary',0)

#         geo.PhysicalGroup.set_group_mesh(True)
#         model.mesh.generate(1)
#         model.mesh.generate(2)
#         gmsh.write(f"{self.name}.msh")
#         os.system(f"gmsh {self.name}.msh &")
