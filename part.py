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


class FenicsPart(object):
    pass

class Fenics2DRVE(FenicsPart):
    """
    Contrat : Créer un couple maillage + matériaux pour des RVE 2D, plans, comportant au plus 2 matériaux constitutifs et pouvant contenir plusieurs cellules.
    """
    def __init__(self, mesh, generating_vectors, material_dict, subdomains, facet_regions):
        """
        Parameters
        ----------
        subdomains : dolfin.MeshFunction
            Indicates the subdomains that have been defined in the mesh.
        #! L'ordre des facet function à probablement de l'importance pour la suite des opérations
        """
        self.mesh = mesh
        self.gen_vect = generating_vectors
        self.rve_area = np.linalg.det(self.gen_vect)
        self.mat_area =  fe.assemble(fe.Constant(1)*fe.dx(mesh))
        self.mesh_dim = mesh.topology().dim() #dimension d'espace de depart
        self.materials = material_dict
        self.subdomains = subdomains
        self.facet_regions = facet_regions
    @staticmethod
    def gmsh_2_Fenics_2DRVE(gmsh_2D_RVE, material_dict, plots=True):
        """
        Generate an instance of Fenics2DRVE from a instance of the Gmsh2DRVE class.

        """
        run(f'dolfin-convert {gmsh_2D_RVE.mesh_abs_path} {gmsh_2D_RVE.name}.xml', shell=True, check=True)
        mesh = fe.Mesh(f'{gmsh_2D_RVE.name}.xml')
        subdomains = fe.MeshFunction('size_t', mesh, f'{gmsh_2D_RVE.name}_physical_region.xml')
        facets = fe.MeshFunction('size_t', mesh, f'{gmsh_2D_RVE.name}_facet_region.xml')
        
        subdo_val = fetools.get_MeshFunction_val(subdomains)
        facets_val = fetools.get_MeshFunction_val(facets)
        logger.info(f'{subdo_val[0]} physical regions imported. The values of their tags are : {subdo_val[1]}')
        logger.info(f'{facets_val[0]} facet regions imported. The values of their tags are : {facets_val[1]}')
        if plots:
            plt.ion()
        plt.figure()
        subdo_plt = fe.plot(subdomains)
        plt.colorbar(subdo_plt)
        plt.figure()
            cmap = plt.cm.get_cmap('viridis', max(facets_val[1])-min(facets_val[1]))
            facets_plt = fetools.facet_plot2d(facets, mesh, cmap=cmap)
        clrbar = plt.colorbar(facets_plt[0])
        # clrbar.set_ticks(facets_val)
            # plt.show()
        logger.info(f'Import of the mesh : DONE')
        
        generating_vectors = gmsh_2D_RVE.gen_vect
        return Fenics2DRVE(mesh, generating_vectors, material_dict, subdomains, facets)





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


    gmsh.option.setNumber('Mesh.SurfaceFaces',1) #Display faces of surface mesh?
    gmsh.fltk.run()

# msh.set_background_mesh(field)

#         gmsh.option.setNumber('Mesh.CharacteristicLengthExtendFromBoundary',0)

#         geo.PhysicalGroup.set_group_mesh(True)
#         model.mesh.generate(1)
#         model.mesh.generate(2)
#         gmsh.write(f"{self.name}.msh")
#         os.system(f"gmsh {self.name}.msh &")
