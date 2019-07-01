# coding: utf8
"""
Created on 18/04/2018
@author: baptiste

"""
import logging
from pathlib import Path

import dolfin as fe
import gmsh

import ho_homog.geometry as geo
from ho_homog import GEO_TOLERANCE, pckg_logger
from ho_homog.mesh_generate_2D import Gmsh2DRVE
from ho_homog.full_scale_pb import FullScaleModel
from ho_homog.materials import Material
from ho_homog.part import FenicsPart
from ho_homog.periodicity import PeriodicDomain
from ho_homog.toolbox_gmsh import process_gmsh_log

logger = logging.getLogger("demo_full_scale")
logger_root = logging.getLogger()
logger.setLevel(logging.INFO)
logger_root.setLevel(logging.INFO)

for hdlr in pckg_logger.handlers[:]:
    if isinstance(hdlr, logging.StreamHandler):
        pckg_logger.removeHandler(hdlr)
pckg_logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s", "%H:%M:%S"
)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger_root.addHandler(stream_handler)


# * Step 1 : Modeling the geometry of the part
geo.init_geo_tools()
a = 1
b, k = a, a / 3
r = 0.05
gmsh.logger.start()
panto_geo = Gmsh2DRVE.pantograph(
    a, b, k, 0.1, nb_cells=(10, 1), soft_mat=True, name="panto_with_soft"
)
process_gmsh_log(gmsh.logger.get())
gmsh.logger.stop()


# * Step 2 : Generating the mesh
lc_ratio = 1 / 3
d_min_max = (2 * r * a, a)
lc_min_max = (lc_ratio * r * a, lc_ratio * a)
panto_geo.main_mesh_refinement((0.1, 0.5), (0.1, 0.3), False)
panto_geo.soft_mesh_refinement((0.1, 0.5), (0.1, 0.3), False)
gmsh.logger.start()
panto_geo.mesh_generate()
process_gmsh_log(gmsh.logger.get())
gmsh.logger.stop()


# * Step 2 : Defining the material properties
E1, nu1 = 1.0, 0.3
E2, nu2 = E1 / 100.0, nu1
E_nu_tuples = [(E1, nu1), (E2, nu2)]

subdo_tags = tuple([subdo.tag for subdo in panto_geo.phy_surf])
material_dict = dict()
for coeff, tag in zip(E_nu_tuples, subdo_tags):
    material_dict[tag] = Material(coeff[0], coeff[1], "cp")

# * Step 3 : Create part object
panto_part = FenicsPart.file_2_FenicsPart(
    panto_geo.mesh_abs_path, material_dict, panto_geo.gen_vect, subdomains_import=True
)
LX = panto_part.global_dimensions[0, 0]


# * Step 4 : Defining boundary conditions
class LeftBorder(fe.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and x[0] <= GEO_TOLERANCE


class RightBorder(fe.SubDomain):
    def inside(self, x, on_boundary):
        right = x[0] >= LX - GEO_TOLERANCE
        return on_boundary and right


l_border = LeftBorder()
r_border = RightBorder()
boundary_markers = fe.MeshFunction("size_t", panto_part.mesh, dim=panto_part.dim - 1)
boundary_markers.set_all(9999)
l_border.mark(boundary_markers, 11)
r_border.mark(boundary_markers, 99)

pbc = PeriodicDomain.pbc_dual_base(panto_part.global_dimensions, "Y")
boundary_conditions = [
    {"type": "Periodic", "constraint": pbc},
    {
        "type": "Dirichlet",
        "constraint": (0.0, 0.0),
        "facet_function": boundary_markers,
        "facet_idx": 11,
    },
    ("Dirichlet", (0.0, 0.0), boundary_markers, 99),
]

# * Step 5 : Defining the load
indicator_fctn = fe.Expression(
    "x[0] >= (LX-Lx_cell)/2.0 && x[0] <= (LX+Lx_cell)/2.0 ? 1 : 0",
    degree=0,
    LX=LX,
    Lx_cell=4.0 * a,
)
load_area = fe.assemble(indicator_fctn * fe.dx(panto_part.mesh))
load_magnitude = 1.0
s_load = load_magnitude / load_area
s_load = fe.Constant((0.0, s_load))  # forme utilisée dans tuto FEniCS linear elasticity
load = (2, s_load, indicator_fctn)

# * Step 6 : Choosing FE characteristics
element = ("Lagrange", 2)

# * Step 7 : Gathering all data in a model
model = FullScaleModel(panto_part, [load], boundary_conditions, element)
model.set_solver("mumps")

# * Step 8 : Solving problem
model.solve(results_file=Path("demo_results/demo_pantograph_full_scale.xdmf"))
