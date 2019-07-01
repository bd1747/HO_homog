# coding: utf-8
"""
Created on 09/01/2019
@author: baptiste

"""

from ho_homog import materials as mat
from ho_homog import part
from ho_homog import mesh_generate_2D
import matplotlib.pyplot as plt
from ho_homog import homog2d as hom
import dolfin as fe
from ho_homog import geometry as geo
import numpy as np


geo.init_geo_tools()

geo.set_gmsh_option("General.Verbosity", 2)
fe.set_log_level(20)
# * CRITICAL  = 50, // errors that may lead to data corruption and suchlike
# * ERROR     = 40, // things that go boom
# * WARNING   = 30, // things that may go boom later
# * INFO      = 20, // information of general interest (default)
# * PROGRESS  = 16, // what's happening (broadly)
# * TRACE     = 13, // what's happening (in detail)
# * DBG       = 10  // sundry
# * https://fenicsproject.org/qa/810/how-to-disable-message-solving-linear-variational-problem/ #noqa


# * Step 1 : Generating the mesh file
a = 1
b, k = a, a / 3
r = a / 1e3
panto_test = mesh_generate_2D.Gmsh2DRVE.pantograph(
    a, b, k, r, nb_cells=(1, 1), soft_mat=True, name="panto_with_soft"
)
panto_test.main_mesh_refinement((3 * r, a / 2), (r / 6, a / 6), True)
panto_test.soft_mesh_refinement((3 * r, a / 2), (r / 6, a / 6), True)
panto_test.mesh_generate()

# * Step 2 : Defining the material mechanical properties for each subdomain
E1, nu1 = 1.0, 0.3
E2, nu2 = E1 / 100.0, nu1
E_nu_tuples = [(E1, nu1), (E2, nu2)]

subdo_tags = tuple(
    [subdo.tag for subdo in panto_test.phy_surf]
)  # Here: soft_mat = True => tags = (1, 2)
material_dict = dict()
for coeff, tag in zip(E_nu_tuples, subdo_tags):
    material_dict[tag] = mat.Material(coeff[0], coeff[1], "cp")

# * Step 3 : Creating the Python object that represents the RVE and is suitable for FEniCS
# * Two alternatives :
# * Step 3.1 : Conversion of the Gmsh2DRVE instance
rve = part.Fenics2DRVE.gmsh_2_Fenics_2DRVE(panto_test, material_dict)

# ! OR:

# * Step 3.2 : Initialization of the Fenics2DRVE instance from
# *             a mesh file + generating vectors
# mesh_path = panto_test.mesh_abs_path
# gen_vect = panto_test.gen_vect #or explicitely : np.array([[4., 0.], [0., 8.]])
# rve = part.Fenics2DRVE.file_2_Fenics_2DRVE(mesh_path, gen_vect, material_dict)

# * Step 4 : Initializing the homogemization model
hom_model = hom.Fenics2DHomogenization(rve)

# * Step 5 : Computing the homogenized consitutive tensors
*localization_dicts, constitutive_tensors = hom_model.homogenizationScheme("EG")

# * Step 6 : Postprocessing
print(constitutive_tensors)
print(constitutive_tensors["E"]["E"])
# *[[0.041  0.0156 0.    ]
# * [0.0156 0.0688 0.    ]
# * [0.     0.     0.0307]]

print(constitutive_tensors["EGbis"]["EGbis"])
# * [[ 0.2831  0.078   0.      0.      0.      0.0336]
# *  [ 0.078   0.0664 -0.      0.     -0.      0.0282]
# *  [ 0.     -0.      0.0756  0.0343  0.0243 -0.    ]
# *  [ 0.      0.      0.0343  0.2289  0.1113  0.    ]
# *  [ 0.     -0.      0.0243  0.1113  0.1419  0.    ]
# *  [ 0.0336  0.0282 -0.      0.      0.      0.0541]]

plt.figure()
fe.plot(
    fe.project(0.1 * hom_model.localization["E"]["U"][2], hom_model.V),
    mode="displacement",
)
plt.savefig("loc_E12_u.pdf")

plt.show()
