# coding: utf-8
"""
Created on 09/01/2019
@author: baptiste

"""

from ho_homog import materials as mat
from ho_homog import part, toolbox_FEniCS
from ho_homog import homog2d as hom
import dolfin as fe
from pathlib import Path
import numpy as np
import logging

# * Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s", "%H:%M")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


fe.set_log_level(20)

GAMMA = 1e-2

# * Step 2 : Defining the material mechanical properties for each subdomain
E1, nu1 = 1.0, 0.3
E2, nu2 = E1 / 100.0, nu1
E_nu_tuples = [(E1, nu1), (E2, nu2)]

subdo_tags = (1, 2)
material_dict = dict()
for coeff, tag in zip(E_nu_tuples, subdo_tags):
    material_dict[tag] = mat.Material(coeff[0], coeff[1], "cp")
gen_vect = np.array([[4.0, 0.0], [0.0, 8.0]])

case = Path("panto_fine.xdmf")
rve = part.Fenics2DRVE.file_2_Fenics_2DRVE(case.with_suffix(".msh"), gen_vect, material_dict)
model = hom.Fenics2DHomogenization(rve)

sigma_fspace = model.W
sigma_loc = list()

scalar_FE = fe.FiniteElement("Lagrange", rve.mesh.ufl_cell(), 2)
scalar_fspace = fe.FunctionSpace(rve.mesh, scalar_FE)

loc_root = case.parent.joinpath(case.stem + "_loc")


for i in range(3):
    loc_path = loc_root.joinpath(f"loc_E{i}_sig.xdmf")
    loc_function = toolbox_FEniCS.function_from_xdmf(
        sigma_fspace, loc_path.stem, loc_path
    )
    sigma_loc.append(loc_function)


nu = mat.ElasticityParamOnSubdomain(rve.subdomains, material_dict, "nu", degree=0)
E = mat.ElasticityParamOnSubdomain(rve.subdomains, material_dict, "E", degree=0)

alpha = (1 + nu) / (1 - nu)
beta = (3 - nu) / (1 + nu)

pre_fact = -1 * 1 / E * (1 - GAMMA) / (1 + alpha * GAMMA)
id_fact = 4
tr_fact = -1 * (1 - GAMMA * (alpha - 2 * beta)) / (1 + beta * GAMMA)

operator_trace = fe.as_vector((1.0, 1.0, 0.0))


def trace_Voigt(eps):
    return fe.inner(operator_trace, eps)


deriv_topo = [[None for j in range(3)] for i in range(3)]
for i in range(3):
    s_i = sigma_loc[i]
    for j in range(i, 3):
        s_j = sigma_loc[j]
        field = pre_fact * (
            id_fact * fe.inner(s_i, s_j) + tr_fact * trace_Voigt(s_i) * trace_Voigt(s_j)
        )
        deriv_topo[i][j] = fe.project(field, scalar_fspace)


results = case.with_name(f"{case.stem}_deriv_topo.xdmf")
with fe.XDMFFile(str(results)) as f_out:
    f_out.parameters["flush_output"] = True
    f_out.parameters["functions_share_mesh"] = True
for i in range(3):
    for j in range(i, 3):
        field = deriv_topo[i][j]
        name = f"deriv_topo_{i}{j}"
        field.rename(name, name)
        f_out.write(field, 0.0)
