# -*- coding: utf-8 -*-
"""
Created on 07/11/2018
@author: baptiste

"""

import geometry as geo
import mesh_tools as msh
import numpy as np
import logging
from logging.handlers import RotatingFileHandler
import gmsh
import os
import matplotlib.pyplot as plt

# nice shortcuts
model = gmsh.model
factory = model.occ

logger = logging.getLogger() #http://sametmax.com/ecrire-des-logs-en-python/
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s') # Afficher le temps à chaque message
file_handler = RotatingFileHandler('activity.log', 'a', 1000000, 1)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler) #Pour écriture d'un fichier log
formatter = logging.Formatter('%(levelname)s :: %(message)s') 
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler) #Pour écriture du log dans la console

prev_wd = os.getcwd()
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
cwd = os.getcwd()
logger.info(f"initial working directory {prev_wd} has been changed to {cwd}")

geo.init_geo_tools()
name = "pantograph"
model.add(name)

logger.info('Start defining the pantograph geometry')
a = 1
b = 1
k = 0.3
r = a/20

Lx = 4*a
Ly = 6*a+2*b
gen_vect = np.array(((Lx,0.), (0.,Ly)))

e1 = np.array((a, 0., 0.))
e2 = np.array((0., a, 0.))
p = np.array((k, 0., 0.))
b = b/a*e2

E1 = geo.Point(e1)
E2 = geo.Point(e2)
E1m = geo.Point(-1*e1)
E2m = geo.Point(-1*e2)
O = np.zeros((3,))
L = geo.Point(2*(e1+e2))
Lm = geo.Point(2*(e1-e2))
M = geo.Point(e1 + 1.5*e2 + b/2)
I = geo.Point(2*(e1 + 1.5*e2 + b/2))

contours = list()
contours.append([E1, E2, E1m, E2m])
contours.append([E1, Lm, geo.Point(3*e1), L])
contours.append([E2, L, geo.translation(L,b/2-p), geo.translation(L,b), geo.translation(E2,b), geo.translation(E2,b/2+p)])
pattern_ll = [geo.LineLoop(pt_list, explicit=False) for pt_list in contours]

pattern_ll += [geo.point_reflection(ll, M) for ll in pattern_ll]
pattern_ll += [geo.plane_reflection(ll, I, e1) for ll in pattern_ll]
pattern_ll += [geo.plane_reflection(ll, I, e2) for ll in pattern_ll]
pattern_ll = geo.remove_duplicates(pattern_ll)
logger.info('Done removing of the line-loops duplicates')

for ll in pattern_ll:
    ll.round_corner_incircle(r)
logger.info('Done rounding all corners of pattern line-loops')

macro_vtcs = [O, gen_vect[0], gen_vect[0] + gen_vect[1] , gen_vect[1]]
macro_ll = geo.LineLoop([geo.Point(c) for c in macro_vtcs])
macro_s = geo.PlaneSurface(macro_ll)

logger.info('Start boolean operations on surfaces')
pattern_s = [geo.PlaneSurface(ll) for ll in pattern_ll]
rve_s = geo.bool_cut_S(macro_s, pattern_s)
rve_s = rve_s[0]
logger.info('Done boolean operations on surfaces')
rve_s_phy = geo.PhysicalGroup([rve_s], 2, "partition_plein")
print(rve_s_phy.__dict__) #!DEBUG
#* >>> rve_s_phy.__dict__
#* {'entities': [<geometry.AbstractSurface object at 0x7fb7aea3a278>], 'dim': 2, 'name': 'partition_plein', 'tag': None}
#* >>> rve_s_phy.entities[0].__dict__
#* {'tag': 26, 'boundary': []}
print(model.getEntities()) #!DEBUG
#* >>> model.getEntities()
#* []
factory.synchronize()
print(model.getEntities(2)) #!DEBUG
#* >>> model.getEntities(2)
#* [(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (2, 9), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (2, 15), (2, 16), (2, 17), (2, 18), (2, 19), (2, 20)]
rve_s_phy.add_gmsh()
factory.synchronize()
#* >>> model.getPhysicalGroups()
print(model.getPhysicalGroups()) #!DEBUG
#* [(2, 1)]
data = model.getPhysicalGroups()
logger.info('All physical groups in the model ' + repr(data)
            + ' Names : ' + repr([model.getPhysicalName(*dimtag) for dimtag in data]))
logger.info('Done generating the gmsh geometrical model')
gmsh.write(f"{name}.brep")
os.system(f"gmsh {dname}/{name}.brep &") #!SUPPR le temps de debug

logger.info('Start defining a mesh refinement constraint')
constr_pts = [pt for ll in pattern_ll for pt in ll.vertices]
fine_pts = [pt for pt in constr_pts if (pt.coord[0] % 1 < p[0]/2. or pt.coord[0] % 1 > 1. - p[0]/2.)]
fine_pts = geo.remove_duplicates(fine_pts)
# #? Test physical groups with points
fine_pts_phy = geo.PhysicalGroup(fine_pts, 0, "fine_points")

# lines_phy = geo.PhysicalGroup(pattern_ll[0].sides, 1, "test_line_phy")
# factory.synchronize()
# fine_pts_phy.add_gmsh()
# lines_phy.add_gmsh()
# logging.info(f"geo.PhysicalGroup.all_groups : {geo.PhysicalGroup.all_groups}")
# factory.synchronize()
# data = model.getPhysicalGroups()
# logger.info('All physical groups in the model ' + repr(data)
#             + ' Names : ' + repr([model.getPhysicalName(*dimtag) for dimtag in data]))
#! POURQUOI CA NE FONCTIONNE PAS ????????!!!!!!! >:(
#TODO : deux problèmes : 1 : avec auxetic_square, le raffinement ne fonctionne pas. 2 : les groupes de Points créés ici et dans create_mesh ne sont JAMAIS présents dans le model geometrique.
#? Est ce que c'est le tri (remove duplicates) d'une liste de points pris sur une liste de courbes qui n'est pas encore triée et qui est triée avant d'être ajoutée à gmsh ?
#* >>> model.getEntities(0)
#* [(0, 1), ...... (0, 278), (0, 279), (0, 280), (0, 281), (0, 282), (0, 283), (0, 284), (0, 285), (0, 286), (0, 287), (0, 288), (0, 289), (0, 290), (0, 291), (0, 292), (0, 293), (0, 294), (0, 295), (0, 296), (0, 297), (0, 298), (0, 299), (0, 300), (0, 301), (0, 302), (0, 303), (0, 304), (0, 305), (0, 306), (0, 307), (0, 308), (0, 309), (0, 310), (0, 311), (0, 312), (0, 313), (0, 314)]
#* >>> factory.synchronize()
#* >>> model.getPhysicalGroups()
#* [(2, 1)]
#* >>> model.getPhysicalGroups(0)
#* []
#! Là c'est un problème
#* >>> model.addPhysicalGroup(0, [311,312,313,314])
#* >>> model.getPhysicalGroups()
#* [(0, 2), (0, 3), (2, 1)]
#! En ajoutant "à la main" un autre physical group, le précédent physical group devient visible
#? CONCLUSION : J'ai l'impression qu'à chaque fois l'ajout d'un second group de la même dimension géométrique est nécessaire pour que les groupes géométriques de cette dimension apparaissent lors d'un appel à model.getPhysicalGroups.
#* >>> test_pts_gp = geo.PhysicalGroup(fine_pts[:5], 0, "point_gp_test")
#* >>> test_pts_gp.__dict__
#* {'entities': [<geometry.Point object at 0x7ff231d16f98>, <geometry.Point object at 0x7ff23216a080>, <geometry.Point object at 0x7ff23216a0f0>, <geometry.Point object at 0x7ff23216a160>, <geometry.Point object at 0x7ff23216a208>], 'dim': 0, 'name': 'point_gp_test', 'tag': None}
#* >>> test_pts_gp.add_gmsh()
#* Physical group 4 of dim 0 add to gmsh
#* >>> model.getPhysicalGroups()
#* [(0, 2), (0, 3), (0, 4), (2, 1)]
#* >>> lines_phy = geo.PhysicalGroup(pattern_ll[0].sides, 1, "test_line_phy")
#* >>> lines_phy.__dict__
#* {'entities': [<geometry.Line object at 0x7ff232175048>, <geometry.Arc object at 0x7ff23216ffd0>, <geometry.Line object at 0x7ff2321752e8>, <geometry.Arc object at 0x7ff2321752b0>, <geometry.Line object at 0x7ff232175588>, <geometry.Arc object at 0x7ff232175550>, <geometry.Line object at 0x7ff232175828>, <geometry.Arc object at 0x7ff2321757f0>], 'dim': 1, 'name': 'test_line_phy', 'tag': None}
#* >>> model.getPhysicalGroups()
#* [(0, 2), (0, 3), (0, 4), (2, 1)]
#* >>> model.getPhysicalGroups()
#* [(0, 2), (0, 3), (0, 4), (2, 1)]
#* >>> lines_phy.add_gmsh()
#* Physical group 5 of dim 1 add to gmsh
#* >>> model.getPhysicalGroups()
#* [(0, 2), (0, 3), (0, 4), (1, 5), (2, 1)]
#? Cette fois si, aucun problème

f = msh.set_mesh_refinement([3*r, a], [r/4., a], attractors={'points':fine_pts}, sigmoid_interpol=False)
msh.set_background_mesh(f)
logger.info('Done defining a mesh refinement constraint')
# macro_bndry = macro_ll.sides #* Bloc de periodicité fonctionne
# rve_s.get_boundary(recursive=True)
# micro_bndry = [geo.macro_line_fragments(rve_s.boundary, M_ln) for M_ln in macro_bndry]
# for i, crvs in enumerate(micro_bndry):
#     msh.order_curves(crvs,
#                     macro_bndry[i%2].def_pts[-1].coord - macro_bndry[i%2].def_pts[0].coord,
#                     orientation=True)
# msh.set_periodicity_pairs(micro_bndry[0], micro_bndry[2])
# msh.set_periodicity_pairs(micro_bndry[1], micro_bndry[3])
# logger.info('Done defining a mesh periodicity constraint')
logging.info(f"Mesh.CharacteristicLengthExtendFromBoundary option : {gmsh.option.getNumber('Mesh.CharacteristicLengthExtendFromBoundary')} Option set to 1")
gmsh.option.setNumber('Mesh.CharacteristicLengthExtendFromBoundary', 1)
gmsh.option.setNumber("Mesh.MeshOnlyVisible", 0)
# geo.PhysicalGroup.set_group_mesh(True)
model.mesh.generate(2)
gmsh.write(f"{name}_refine_points.msh")
os.system(f"gmsh {dname}/{name}_refine_points.msh &")
os.system(f"gmsh {name}_refine_points.msh &")
gmsh.fltk.run()
all_lines = list()
for ll in pattern_ll:
    all_lines += ll.sides
lines_phy = geo.PhysicalGroup(all_lines, 1, "test_line_phy")
factory.synchronize()
lines_phy.add_gmsh()
factory.synchronize()
logger.info(f"Physical groups after After adding lines_phy to the model : {model.getPhysicalGroups()}")

g = msh.set_mesh_refinement([2*r, a], [r/2, a/2], attractors={'curves':lines_phy.entities}, sigmoid_interpol=True)
msh.set_background_mesh(g)
geo.PhysicalGroup.set_group_mesh(True)
model.mesh.generate(2)
gmsh.write(f"{name}_refine_lines.msh")
os.system(f"gmsh {dname}/{name}_refine_lines.msh &")
logging.info(f"Initial value of Mesh.CharacteristicLengthExtendFromBoundary option : {gmsh.option.getNumber('Mesh.CharacteristicLengthExtendFromBoundary')} Option set to 0")
gmsh.option.setNumber('Mesh.CharacteristicLengthExtendFromBoundary',0)
#* OK ! Avec cette valeur d'option ça fonctionne comme voulu !


# fine_pts_phy.add_gmsh()
# lines_phy.add_gmsh()
# logging.info(f"geo.PhysicalGroup.all_groups : {geo.PhysicalGroup.all_groups}")
# factory.synchronize()


# gmsh.fltk.run()