# coding: utf-8
"""
Created on 09/01/2019
@author: baptiste

"""

from ho_homog import materials as mat
from ho_homog import part
from ho_homog import mesh_generate
import matplotlib.pyplot as plt
from ho_homog import homog2d as hom
import dolfin as fe
from ho_homog import geometry as geo
from pathlib import Path

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

with_soft_mat = True  # Use soft material or voids

# * Step 1 : Generating the mesh file
a = 1
b, k = a, a / 3
r = a / 1e3
rve_name = "panto_with_soft" if with_soft_mat else "panto_with_voids"
panto_test = mesh_generate.pantograph.pantograph_RVE(
    a, b, k, r, nb_cells=(1, 1), soft_mat=with_soft_mat, name=rve_name
)
panto_test.main_mesh_refinement((3 * r, a / 2), (r /6, a /6), True)
if with_soft_mat:
    panto_test.soft_mesh_refinement((3 * r, a / 2), (r /6, a /6), True)
panto_test.mesh_generate()

# * Step 2 : Defining the material mechanical properties for each subdomain
E1, nu1 = 1.0, 0.3
E2, nu2 = E1 / 100.0, nu1
E_nu_tuples = [(E1, nu1), (E2, nu2)]


# * Step 3 : Creating the Python object that represents the RVE and is suitable for FEniCS
# * Two alternatives :
# * Step 3.1 : Conversion of the Gmsh2DRVE instance
if with_soft_mat:
    subdo_tags = tuple(
        [subdo.tag for subdo in panto_test.phy_surf]
    )  # Here: soft_mat = True => tags = (1, 2)
    material_dict = dict()
    for coeff, tag in zip(E_nu_tuples, subdo_tags):
        material_dict[tag] = mat.Material(coeff[0], coeff[1], "cp")
    rve = part.Fenics2DRVE.rve_from_gmsh2drve(panto_test, material_dict)
else:
    material = mat.Material(E1, nu1, "cp")
    rve = part.Fenics2DRVE.rve_from_gmsh2drve(panto_test, material)

# ! OR:

# * Step 3.2 : Initialization of the Fenics2DRVE instance from
# *             a mesh file + generating vectors
# mesh_path = Path("panto_with_soft.msh")
# gen_vect = np.array([[4., 0.], [0., 8.]])
# rve = part.Fenics2DRVE.rve_from_file(mesh_path, gen_vect, material_dict)

# * Step 4 : Initializing the homogemization model
hom_model = hom.Fenics2DHomogenization(rve)

# * Step 5 : Computing the homogenized consitutive tensors
*localization_dicts, constitutive_tensors = hom_model.homogenizationScheme("EG")

# * Step 6 : Postprocessing
print(constitutive_tensors)
print(constitutive_tensors["E"]["E"])
# With soft material:
# *[[0.041  0.0156 0.    ]
# * [0.0156 0.0688 0.    ]
# * [0.     0.     0.0307]]

# With voids:
# * [[ 6.348e-05  9.939e-05 -2.753e-11]
# *  [ 9.939e-05  3.052e-02 -2.018e-09]
# *  [-2.753e-11 -2.018e-09  7.747e-05]]

print(constitutive_tensors["EGbis"]["EGbis"])
# With soft material:
# * [[ 0.2831  0.078   0.      0.      0.      0.0336]
# *  [ 0.078   0.0664 -0.      0.     -0.      0.0282]
# *  [ 0.     -0.      0.0756  0.0343  0.0243 -0.    ]
# *  [ 0.      0.      0.0343  0.2289  0.1113  0.    ]
# *  [ 0.     -0.      0.0243  0.1113  0.1419  0.    ]
# *  [ 0.0336  0.0282 -0.      0.      0.      0.0541]]
# With voids:
# * [[ 7.795e-01  4.434e-01 -9.629e-09  5.304e-10    1.432e-07  5.697e-02],
# *  [ 4.434e-01  5.592e-01 -1.054e-06  5.947e-08    3.943e-08  2.206e-01],
# *  [-9.629e-09 -1.054e-06  1.076e+00  4.861e-02   1.893e-01  7.280e-09],
# *  [ 5.304e-10  5.947e-08  4.861e-02  6.779e-01   6.012e-01 -8.488e-09],
# *  [ 1.432e-07  3.943e-08  1.893e-01  6.012e-01   8.900e-01 -3.776e-07],
# *  [ 5.697e-02  2.206e-01  7.280e-09 -8.488e-09  -3.776e-07  4.015e-01]]


plt.figure()
fe.plot(
    fe.project(0.1 * hom_model.localization["E"]["U"][2], hom_model.V),
    mode="displacement",
)
# plt.savefig("loc_E12_u.pdf")

plt.show()
