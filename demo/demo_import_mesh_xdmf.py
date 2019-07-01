# coding: utf-8
"""
Created on 01/07/2019
@author: baptiste

1- Conversion of a mesh file MSH format (version 4) -> XDMF format,
2- Import of the mesh from the XDMF file,
3- Show facet regions and subdomains with matplotlib.
"""

import logging

import gmsh
import matplotlib.pyplot as plt
import dolfin as fe
from ho_homog import geometry, mesh_generate_2D, toolbox_FEniCS, toolbox_gmsh

# * Logging
logger = logging.getLogger(__name__)  # http://sametmax.com/ecrire-des-logs-en-python/
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s", "%H:%M")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

geometry.init_geo_tools()
geometry.set_gmsh_option("Mesh.MshFileVersion", 4.1)

logger.info("Generating the geometry model")
a = 1
b, k = a, a / 3
r = 0.01 * a
rve_geo = mesh_generate_2D.Gmsh2DRVE.pantograph(a, b, k, r, soft_mat=True, name="panto")

logger.info("Generating the mesh")
lc_ratio = 1 / 3.0
d_min_max = (2 * r, a)
lc_min_max = (lc_ratio * r, lc_ratio * a)
rve_geo.main_mesh_refinement(d_min_max, lc_min_max, False)
rve_geo.soft_mesh_refinement(d_min_max, lc_min_max, False)
rve_geo.mesh_generate()

logger.info("Saving mesh with MSH 4 format")
gmsh.model.mesh.renumberNodes()
gmsh.model.mesh.renumberElements()
gmsh.write(str(rve_geo.mesh_abs_path))

logger.info("Mesh conversion MSH -> XDMF")
mesh_path, *_ = toolbox_gmsh.msh_conversion(
    rve_geo.mesh_abs_path, ".xdmf", subdomains=True
)

logger.info("Import of mesh and partitions as MeshFunction instances")
mesh, subdomains, facet_regions = toolbox_FEniCS.xdmf_mesh(mesh_path, True)


plt.figure()
fe.plot(mesh, title="Mesh only")
plt.figure()
subdo_plt = fe.plot(subdomains, title="Subdomains")
plt.colorbar(subdo_plt)

nb_val, facets_val = toolbox_FEniCS.get_MeshFunction_val(facet_regions)
facets_val = facets_val[facets_val != 18446744073709551615]
facets_val = facets_val[facets_val != 0]
print(facets_val)
delta = max(facets_val) - max(facets_val)
cmap = plt.cm.get_cmap("viridis", max(facets_val) - min(facets_val)+1)
fig, ax = plt.subplots()
facets_plt = toolbox_FEniCS.facet_plot2d(
    facet_regions, mesh, cmap=cmap, exclude_val=(0, 18446744073709551615)
)
ax.set_title("Facet regions")
cbar = fig.colorbar(facets_plt[0], ticks=list(facets_val))
cbar.ax.set_xticklabels(list(facets_val))

plt.show()
