# coding: utf-8
"""
Created on 09/01/2019
@author: baptiste

"""

import materials as mat
import part
import mesh_generate_2D
import matplotlib.pyplot as plt
import homog2d as hom
import dolfin as fe
import geometry as geo
from pathlib import Path
import numpy as np
import logging
from logging.handlers import RotatingFileHandler


geo.init_geo_tools()
plt.ioff()

#* Logging
logger = logging.getLogger() #* Accès au root logger
#source : http://sametmax.com/ecrire-des-logs-en-python/
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s',"%Y-%m-%d %H:%M:%S")
log_path = Path.home().joinpath('Desktop/activity.log')
file_handler = RotatingFileHandler(str(log_path), 'a', 1000000, 2)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler) #Pour écriture d'un fichier log
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s',"%H:%M")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

geo.set_gmsh_option("General.Verbosity", 2)

fe.set_log_level(16)
#* CRITICAL  = 50, // errors that may lead to data corruption and suchlike
#* ERROR     = 40, // things that go boom
#* WARNING   = 30, // things that may go boom later
#* INFO      = 20, // information of general interest (default)
#* PROGRESS  = 16, // what's happening (broadly)
#* TRACE     = 13, // what's happening (in detail)
#* DBG       = 10  // sundry
#* https://fenicsproject.org/qa/810/how-to-disable-message-solving-linear-variational-problem/


#* Step 1 : Generating the mesh file
a = 1
b, k = a, a/3
panto_test = mesh_generate_2D.Gmsh2DRVE.pantograph(a, b, k, 0.1, soft_mat=True, name='panto_with_soft')
panto_test.main_mesh_refinement((0.1,0.5),(0.1,0.3),False)
panto_test.soft_mesh_refinement((0.1,0.5),(0.1,0.3),False)
panto_test.mesh_generate()

#* Step 2 : Defining the material mechanical properties for each subdomain
E1, nu1 = 1., 0.3
E2, nu2 = E1/100., nu1
E_nu_tuples = [(E1, nu1), (E2, nu2)]

subdo_tags = tuple([subdo.tag for subdo in panto_test.phy_surf]) # Here: soft_mat = True => tags = (1, 2)
material_dict = dict()
for coeff, tag in zip(E_nu_tuples, subdo_tags):
    material_dict[tag] = mat.Material(coeff[0], coeff[1], 'cp')

#* Step 3 : Creating the Python object that represents the RVE and is suitable for FEniCS
#* Two alternatives :
#* Step 3.1 : Conversion of the Gmsh2DRVE instance
rve = part.Fenics2DRVE.gmsh_2_Fenics_2DRVE(panto_test, material_dict)

#! OR !#

#* Step 3.2 : Initialization of the Fenics2DRVE instance from a mesh file + generating vectors
# mesh_path = panto_test.mesh_abs_path
# gen_vect = panto_test.gen_vect #or explicitely : np.array([[4., 0.], [0., 8.]])
# rve = part.Fenics2DRVE.file_2_Fenics_2DRVE(mesh_path, gen_vect, material_dict)

#* Step 4 : Initializing the homogemization model
hom_model = hom.Fenics2DHomogenization(rve)

#* Step 5 : Computing the homogenized consitutive tensors
DictOfLocalizationsU, DictOfLocalizationsSigma, DictOfLocalizationsEpsilon, DictOfConstitutiveTensors = hom_model.homogenizationScheme('EG')

#* Step 6 : Postprocessing
print(DictOfConstitutiveTensors)
print(DictOfConstitutiveTensors['E']['E'])
#* [[ 0.0726  0.0379 -0.    ]
#*  [ 0.0379  0.1638  0.    ]
#*  [-0.      0.      0.0906]]

print(DictOfConstitutiveTensors['EGbis']['EGbis'])
#* [[ 0.3799  0.1405 -0.      0.      0.      0.0401]
#*  [ 0.1405  0.14   -0.      0.      0.      0.0451]
#*  [-0.     -0.      0.1428  0.0393  0.039  -0.    ]
#*  [ 0.      0.      0.0393  0.292   0.1822 -0.    ]
#*  [ 0.      0.      0.039   0.1822  0.2401 -0.    ]
#*  [ 0.0401  0.0451 -0.     -0.     -0.      0.0676]]

plt.figure()
fe.plot(fe.project(0.1*hom_model.localization['E']['U'][2],hom_model.V), mode='displacement')
plt.savefig("loc_EU.pdf")

plt.show()