# coding: utf8
"""
Created on 17/11/2018
@author: baptiste

"""

import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
import numpy as np
import ho_homog.toolbox_FEniCS as fetools
import dolfin as fe
import ho_homog.materials as mat
from subprocess import run
from pathlib import Path

plt.ioff()

logger = logging.getLogger(__name__) #http://sametmax.com/ecrire-des-logs-en-python/
logger.setLevel(logging.INFO)


class FenicsPart(object):
    """
    Contrat : Créer un couple maillage + matériaux pour des géométries 2D, planes.
    """
    def __init__(self, mesh, part_vectors, material_dict, subdomains, facet_regions):
        """
        Parameters
        ----------
        subdomains : dolfin.MeshFunction
            Indicates the subdomains that have been defined in the mesh.
        #! L'ordre des facet function à probablement de l'importance pour la suite des opérations
        """
        self.mesh = mesh
        self.gen_vect = part_vectors
        self.rve_area = np.linalg.det(self.gen_vect)
        self.mat_area = fe.assemble(fe.Constant(1)*fe.dx(mesh))
        self.mesh_dim = mesh.topology().dim() #dimension d'espace de depart
        self.materials = material_dict
        self.subdomains = subdomains
        self.facet_regions = facet_regions
    
        self.C_per = mat.mat_per_subdomains(self.subdomains, self.materials, self.mesh_dim)
    #! A REGARDER !
    def epsilon(self,u):
        return mat.epsilon(u)
    def sigma(self,eps):
        return mat.sigma(self.C_per, eps)
    def StrainCrossEnergy(self, sig, eps):
        return mat.strain_cross_energy(sig, eps, self.mesh, self.rve_area)

    @staticmethod
    def file_2_FenicsPart(mesh_path, part_vectors, material_dict, subdomains_import=False, plots=True, explicit_subdo_val=1):
        """Generate an instance of Fenics2DRVE from a .xml or .msh file that contains the mesh.

        Parameters
        ----------
        mesh_path : string or Path
            Relative or absolute path to the mesh file (format Dolfin XML or MSH version 2)
        generating_vectors : 2D-array
            dimensions of the RVE
        material_dict : dictionnary
            [description] #TODO : à compléter
        subdomains_import : bool
            Import subdomains data ?
        plots : bool, optional
            If True (default) the physical regions and the facet regions are plotted at the end of the import.

        Returns
        -------
        Fenics2DRVE instance

        """
        if not isinstance(mesh_path, Path):
            mesh_path = Path(mesh_path)
        name = mesh_path.stem

        if mesh_path.suffix != '.xml':
            cmd = (
                "dolfin-convert "
                + mesh_path.as_posix()
                + ' '
                + mesh_path.with_suffix('.xml').as_posix()
                )
            run(cmd, shell=True, check=True)
            mesh_path = mesh_path.with_suffix('.xml')

        mesh = fe.Mesh(mesh_path.as_posix())
        if subdomains_import:
            subdo_path = mesh_path.with_name(name+'_physical_region.xml')
            facet_path = mesh_path.with_name(name+'_facet_region.xml')
            if subdo_path.exists():
                subdomains = fe.MeshFunction('size_t', mesh, subdo_path.as_posix())
            else:
                logger.info(f"For mesh file {mesh_path.name}, _physical_region.xml file is missing.")
                subdomains = None
            if facet_path.exists():
                facets = fe.MeshFunction('size_t', mesh, facet_path.as_posix())
            else:
                logger.info(f"For mesh file {mesh_path.name}, _facet_region.xml file is missing.")
                facets = None
            subdo_val = fetools.get_MeshFunction_val(subdomains)
            facets_val = fetools.get_MeshFunction_val(facets)
            logger.info(f'{subdo_val[0]} physical regions imported. The values of their tags are : {subdo_val[1]}')
            logger.info(f'{facets_val[0]} facet regions imported. The values of their tags are : {facets_val[1]}')
        else:
            subdomains = fe.MeshFunction('size_t', mesh, mesh.topology().dim())
            subdomains.set_all(explicit_subdo_val)
            facets = None
        
        if plots:
            plt.figure()
            subdo_plt = fe.plot(subdomains)
            plt.colorbar(subdo_plt)
            plt.figure()
            cmap = plt.cm.get_cmap('viridis', max(facets_val[1])-min(facets_val[1]))
            facets_plt = fetools.facet_plot2d(facets, mesh, cmap=cmap)
            plt.colorbar(facets_plt[0])
            plt.draw()
        logger.info(f'Import of the mesh : DONE')
        
        return FenicsPart(mesh, part_vectors, material_dict, subdomains, facets)

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
        self.mat_area = fe.assemble(fe.Constant(1)*fe.dx(mesh))
        self.mesh_dim = mesh.topology().dim() #dimension d'espace de depart
        self.materials = material_dict
        self.subdomains = subdomains
        self.facet_regions = facet_regions
    
        self.C_per = mat.mat_per_subdomains(self.subdomains, self.materials, self.mesh_dim)
    #! A REGARDER !
    def epsilon(self,u):
        return mat.epsilon(u)
    def sigma(self,eps):
        return mat.sigma(self.C_per, eps)
    def StrainCrossEnergy(self, sig, eps):
        return mat.strain_cross_energy(sig, eps, self.mesh, self.rve_area)


    @staticmethod
    def gmsh_2_Fenics_2DRVE(gmsh_2D_RVE, material_dict, plots=True):
        """
        Generate an instance of Fenics2DRVE from a instance of the Gmsh2DRVE class.

        """
        xml_path = gmsh_2D_RVE.mesh_abs_path.with_suffix('.xml')
        cmd = ('dolfin-convert '
                + gmsh_2D_RVE.mesh_abs_path.as_posix()
                + ' '
                + xml_path.as_posix()
                )
        run(cmd, shell=True, check=True)
        mesh = fe.Mesh(xml_path.as_posix())
        name = xml_path.stem
        subdomains = fe.MeshFunction(
                    'size_t',
                    mesh,
                    xml_path.with_name(name + '_physical_region.xml').as_posix()
                    )
        facets = fe.MeshFunction(
                    'size_t',
                    mesh,
                    xml_path.with_name(name + '_facet_region.xml').as_posix()
                    )
        subdo_val = fetools.get_MeshFunction_val(subdomains)
        facets_val = fetools.get_MeshFunction_val(facets)
        logger.info(f'{subdo_val[0]} physical regions imported. The values of their tags are : {subdo_val[1]}')
        logger.info(f'{facets_val[0]} facet regions imported. The values of their tags are : {facets_val[1]}')
        if plots:
            plt.figure()
            subdo_plt = fe.plot(subdomains)
            plt.colorbar(subdo_plt)
            plt.figure()
            cmap = plt.cm.get_cmap('viridis', max(facets_val[1])-min(facets_val[1]))
            facets_plt = fetools.facet_plot2d(facets, mesh, cmap=cmap)
            clrbar = plt.colorbar(facets_plt[0])
            # clrbar.set_ticks(facets_val)
            plt.draw()
        logger.info(f'Import of the mesh : DONE')
        
        generating_vectors = gmsh_2D_RVE.gen_vect
        return Fenics2DRVE(mesh, generating_vectors, material_dict, subdomains, facets)

    @staticmethod
    def file_2_Fenics_2DRVE(mesh_path, generating_vectors, material_dict, plots=True):
        """Generate an instance of Fenics2DRVE from a .xml or .msh file that contains the mesh.

        Parameters
        ----------
        mesh_path : string or Path
            Relative or absolute path to the mesh file (format Dolfin XML or MSH version 2)
        generating_vectors : 2D-array
            dimensions of the RVE
        material_dict : dictionnary
            [description] #TODO : à compléter
        plots : bool, optional
            If True (default) the physical regions and the facet regions are plotted at the end of the import.

        Returns
        -------
        Fenics2DRVE instance

        """
        if not isinstance(mesh_path, Path):
            mesh_path = Path(mesh_path)
        name = mesh_path.stem

        if mesh_path.suffix != '.xml':
            cmd = (
                "dolfin-convert "
                + mesh_path.as_posix()
                + ' '
                + mesh_path.with_suffix('.xml').as_posix()
                )
            run(cmd, shell=True, check=True)
            mesh_path = mesh_path.with_suffix('.xml')

        mesh = fe.Mesh(mesh_path.as_posix())
        subdo_path = mesh_path.with_name(name+'_physical_region.xml')
        facet_path = mesh_path.with_name(name+'_facet_region.xml')
        if subdo_path.exists():
            subdomains = fe.MeshFunction('size_t', mesh, subdo_path.as_posix())
        else:
            logger.info(f"For mesh file {mesh_path.name}, _physical_region.xml file is missing.")
            subdomains = None
        if facet_path.exists():
            facets = fe.MeshFunction('size_t', mesh, facet_path.as_posix())
        else:
            logger.info(f"For mesh file {mesh_path.name}, _facet_region.xml file is missing.")
            facets = None
        
        subdo_val = fetools.get_MeshFunction_val(subdomains)
        facets_val = fetools.get_MeshFunction_val(facets)
        logger.info(f'{subdo_val[0]} physical regions imported. The values of their tags are : {subdo_val[1]}')
        logger.info(f'{facets_val[0]} facet regions imported. The values of their tags are : {facets_val[1]}')
        if plots:
            plt.figure()
            subdo_plt = fe.plot(subdomains)
            plt.colorbar(subdo_plt)
            plt.figure()
            cmap = plt.cm.get_cmap('viridis', max(facets_val[1])-min(facets_val[1]))
            facets_plt = fetools.facet_plot2d(facets, mesh, cmap=cmap)
            plt.colorbar(facets_plt[0])
            plt.draw()
        logger.info(f'Import of the mesh : DONE')
        
        return Fenics2DRVE(mesh, generating_vectors, material_dict, subdomains, facets)

if __name__ == "__main__":
    pass
    # geo.init_geo_tools()

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


    # gmsh.option.setNumber('Mesh.SurfaceFaces',1) #Display faces of surface mesh?
    # gmsh.fltk.run()

# msh.set_background_mesh(field)

#         gmsh.option.setNumber('Mesh.CharacteristicLengthExtendFromBoundary',0)

#         geo.PhysicalGroup.set_group_mesh(True)
#         model.mesh.generate(1)
#         model.mesh.generate(2)
#         gmsh.write(f"{self.name}.msh")
#         os.system(f"gmsh {self.name}.msh &")
