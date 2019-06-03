# coding: utf8
"""
Created on 18/04/2018
@author: baptiste

"""
import logging
from pathlib import Path

import dolfin as fe
import gmsh

from ho_homog import *

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

gmsh_logger = logging.getLogger("gmsh")
gmsh_logger.setLevel(logging.INFO)


def process_gmsh_log(gmsh_log: list, detect_error=True):
    """Treatment of log messages gathered with gmsh.logger.get()"""
    err_msg, warn_msg = list(), list()
    for line in gmsh_log:
        if "error" in line.lower():
            err_msg.append(line)
        if "warning" in line.lower():
            warn_msg.append(line)
    gmsh_logger.info("**********")
    gmsh_logger.info(
        f"{len(gmsh_log)} logging messages got from Gmsh : {len(err_msg)} errors, {len(warn_msg)} warnings."
    )
    if err_msg:
        gmsh_logger.error("Gmsh errors details :")
        for line in err_msg:
            gmsh_logger.error(line)
    if warn_msg:
        gmsh_logger.warning("Gmsh warnings details :")
        for line in warn_msg:
            gmsh_logger.warning(line)
    gmsh_logger.debug("All gmsh messages :")
    gmsh_logger.debug(gmsh_log)
    gmsh_logger.info("**********")
    if detect_error and err_msg:
        raise AssertionError("Gmsh logging messages signal errors.")


# * Step 1 : Modeling the geometry of the part
geometry.init_geo_tools()
a = 1
b, k = a, a / 3
r = 0.05
gmsh.logger.start()
panto_geo = mesh_generate_2D.Gmsh2DRVE.pantograph(
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
    material_dict[tag] = materials.Material(coeff[0], coeff[1], "cp")

# * Step 3 : Create part object
panto_part = part.Fenics2DRVE.file_2_FenicsPart(
    panto_geo.mesh_abs_path,
    material_dict,
    global_dimensions=panto_geo.gen_vect,
    subdomains_import=True,
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
boundary_markers = fe.MeshFunction("size_t", panto_part.mesh, dim=panto_part.dim-1)
boundary_markers.set_all(9999)
l_border.mark(boundary_markers, 11)
r_border.mark(boundary_markers, 99)

pbc = full_scale_pb.PeriodicDomain.pbc_dual_base(panto_part.global_dimensions, "Y")
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
model = full_scale_pb.FullScaleModel(panto_part, [load], boundary_conditions, element)
model.set_solver("LU", "mumps")

# * Step 8 : Solving problem
model.solve(results_file=Path("demo_results/demo_pantograph_full_scale.xdmf"))

