# coding: utf-8
"""
Created on 09/01/2019
@author: baptiste

"""

from ho_homog import mesh_generate_2D, mesh_tools
from ho_homog import geometry as geo
import logging
import gmsh


# * Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s", "%H:%M")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

geo.init_geo_tools()

geo.set_gmsh_option("General.Verbosity", 2)
geo.set_gmsh_option("Mesh.MshFileVersion", 4.1)
geo.set_gmsh_option("Mesh.Algorithm", 6)


# * Step 1 : Generating the mesh file
a = 1
b, k = a, a / 3

panto_coarse = mesh_generate_2D.Gmsh2DRVE.pantograph(
    a, b, k, 0.1, nb_cells=(1, 1), soft_mat=True, name="panto_coarse"
)
panto_coarse.main_mesh_refinement((0.2, 0.5), (0.05, 0.3), True)
panto_coarse.soft_mesh_refinement((0.2, 0.5), (0.05, 0.3), True)
panto_coarse.mesh_generate()
gmsh.model.mesh.renumberNodes()
gmsh.model.mesh.renumberElements()
gmsh.write(str(panto_coarse.mesh_abs_path))

panto_fine = mesh_generate_2D.Gmsh2DRVE.pantograph(
    a, b, k, 0.1, nb_cells=(1, 1), soft_mat=True, name="panto_fine"
)
panto_fine.main_mesh_refinement((0.2, 0.5), (0.025, 0.1), True)
panto_fine.soft_mesh_refinement((0.2, 0.5), (0.025, 0.1), True)
panto_fine.mesh_generate()
gmsh.model.mesh.renumberNodes()
gmsh.model.mesh.renumberElements()
gmsh.write(str(panto_fine.mesh_abs_path))

# * Step 1 : Conversion of the mesh files
mesh_tools.msh_conversion("panto_coarse.msh")
mesh_tools.msh_conversion("panto_fine.msh")
