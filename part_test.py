# coding: utf-8
"""
Created on 09/01/2019
@author: baptiste

"""

import geometry as geo
import part as prt
from subprocess import run
import materials as mat
import logging
from logging.handlers import RotatingFileHandler
import gmsh

logger = logging.getLogger(__name__) #http://sametmax.com/ecrire-des-logs-en-python/
logger.setLevel(logging.INFO)
if __name__ == "__main__":
    logger_root = logging.getLogger()
    logger_root.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s') # Afficher le temps à chaque message
file_handler = RotatingFileHandler(f'activity_{__name__}.log', 'a', 1000000)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
    logger_root.addHandler(file_handler) #Pour écriture d'un fichier log
formatter = logging.Formatter('%(levelname)s :: %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
    logger_root.addHandler(stream_handler)

geo.init_geo_tools()

geo.set_gmsh_option("Mesh.SaveAll", 0)
geo.set_gmsh_option("Mesh.Binary", 0)
geo.set_gmsh_option("Mesh.MshFileVersion", 2.2)

#? Test du constructeur gmsh_2_Fenics_2DRVE
# a = 1
# b, k = a, a/3
# panto_test = prt.Gmsh2DRVE.pantograph(a, b, k, 0.1, nb_cells=(2, 3), soft_mat=False, name='panto_test')
# panto_test.main_mesh_refinement((0.1,0.5),(0.03,0.3),False)

# panto_test.mesh_generate()
# run(f"gmsh {panto_test.name}.msh &",shell=True, check=True)

# E1, nu1 = 1., 0.3
# E2, nu2 = E1/100., nu1
# E_nu_tuples = [(E1, nu1), (E2, nu2)]
# phy_subdomains = panto_test.phy_surf
# material_dict = dict()
# for coeff, subdo in zip(E_nu_tuples, phy_subdomains):
#     material_dict[subdo.tag] = mat.Material(coeff[0], coeff[1], 'cp')
# rve = prt.Fenics2DRVE.gmsh_2_Fenics_2DRVE(panto_test, material_dict)
#* Test ok

#? Problème de maillage lorsqu'un domaine mou existe
# a = 1
# b, k, r = a, a/3, a/10
# panto_test = prt.Gmsh2DRVE.pantograph(a, b, k, r, soft_mat=True, name='panto_test_soft')
# run(f"gmsh {panto_test.name}.brep &", shell=True, check=True)
# panto_test.main_mesh_refinement((r, 5*r), (r/2, a/3), False)
# panto_test.soft_mesh_refinement((r, 5*r), (r/2, a/3), False)
# panto_test.mesh_generate() #!Sans l'opération de removeDuplicateNodes() dans mesh_generate(). Avec domaine mou obtenu par intersection.
# gmsh.model.mesh.removeDuplicateNodes()
# gmsh.write(f"{panto_test.name}_clean.msh")
# run(f"gmsh {panto_test.name}.msh &", shell=True, check=True)
# run(f"gmsh {panto_test.name}_clean.msh &", shell=True, check=True)
# a = 1
# b, k, r = a, a/3, a/10
# panto_test = prt.Gmsh2DRVE.pantograph(a, b, k, r, soft_mat=True, name='panto_test_soft')
# run(f"gmsh {panto_test.name}.brep &", shell=True, check=True)
# panto_test.main_mesh_refinement((r, 5*r), (r/2, a/3), False)
# panto_test.soft_mesh_refinement((r, 5*r), (2*r, a/3), False)
# panto_test.mesh_generate() #!Sans l'opération de removeDuplicateNodes() dans mesh_generate(). Avec domaine mou obtenu par intersection.
# gmsh.write(f"{panto_test.name}_clean.msh")
# run(f"gmsh {panto_test.name}.msh &", shell=True, check=True)
# run(f"gmsh {panto_test.name}_clean.msh &", shell=True, check=True)
#? Fin de la résolution


#? Test d'un mesh avec un domaine mou, gmsh2DRVE puis import dans FEniCS
a = 1
b, k = a, a/3
panto_test = prt.Gmsh2DRVE.pantograph(a, b, k, 0.1, nb_cells=(2, 3), soft_mat=True, name='panto_test_mou')
panto_test.main_mesh_refinement((0.1,0.5),(0.03,0.3),False)
panto_test.soft_mesh_refinement((0.1,0.5),(0.03,0.3),False)
panto_test.mesh_generate()
run(f"gmsh {panto_test.name}.msh &",shell=True, check=True)
E1, nu1 = 1., 0.3
E2, nu2 = E1/100., nu1
E_nu_tuples = [(E1, nu1), (E2, nu2)]
phy_subdomains = panto_test.phy_surf
material_dict = dict()
for coeff, subdo in zip(E_nu_tuples, phy_subdomains):
    material_dict[subdo.tag] = mat.Material(coeff[0], coeff[1], 'cp')
rve = prt.Fenics2DRVE.gmsh_2_Fenics_2DRVE(panto_test, material_dict)
pass
#* Test ok