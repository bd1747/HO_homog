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

# nice shortcuts
model = gmsh.model
factory = model.occ

logger = logging.getLogger() #http://sametmax.com/ecrire-des-logs-en-python/
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s') # Afficher le temps à chaque message
file_handler = RotatingFileHandler('activity.log', 'a', 1000000, 1)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler) #Pour écriture d'un fichier log
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler) #Pour écriture du log dans la console

geo.init_geo_tools()
name = "pantograph"
model.add(name)

logger.info('Start defining the pantograph geometry')
a = 1
b = 1
k = 0.3

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
sym_ll = [geo.plane_reflection(ll, I, e1) for ll in pattern_ll]
for ll in sym_ll:
    ll.reverse()
pattern_ll += sym_ll
sym_ll = [geo.plane_reflection(ll, I, e2) for ll in pattern_ll]
for ll in sym_ll:
    ll.reverse()
pattern_ll += sym_ll
pattern_ll = geo.remove_duplicates(pattern_ll)
logger.info(f"Done removing of the line-loops duplicates. pattern_ll length : {len(pattern_ll)}")
# 13 no-redundant LineLoop to define the pantographe microstructure geometry in one cell.

r = a/20
for ll in pattern_ll:
    ll.round_corner_incircle(r)
logger.info('Done rounding all corners of pattern line-loops')

macro_vtcs = [O, gen_vect[0], gen_vect[0] + gen_vect[1] , gen_vect[1]]
macro_ll = geo.LineLoop([geo.Point(c) for c in macro_vtcs])
macro_s = geo.PlaneSurface(macro_ll)

logger.info('Start boolean operations on surfaces')
pattern_s = [geo.PlaneSurface(ll) for ll in pattern_ll]
rve_s = geo.AbstractSurface.bool_cut(macro_s, pattern_s)
rve_s = rve_s[0]
logger.info('Done boolean operations on surfaces')
rve_s_phy = geo.PhysicalGroup([rve_s], 2, "partition_plein")
# logger.info('Add all the required geometrical entities to the geometrical model')

# #Suppression à la main des outils...
# data = gmsh.model.getEntities(2)
# # data += gmsh.model.getEntities(1)
# logger.debug('surfaces and curves before remove : ' + repr(data))
# # factory.remove([(2, s.tag) for s in pattern_s] + [(2,macro_s.tag)])
# factory.remove([(2, s.tag) for s in pattern_s])
# factory.remove([(1, l.tag) for ll in pattern_ll for l in ll.sides])
# for s in pattern_s:
#     for  l in s.ext_contour.sides:
#         l.tag = None
#     s.tag = None
# macro_s.tag = None
# factory.synchronize()
# data = gmsh.model.getEntities(2)
# # data += gmsh.model.getEntities(1)
# logger.debug('surfaces and curves after remove : ' + repr(data))
factory.synchronize()
rve_s_phy.add_gmsh()
factory.synchronize()
data = model.getPhysicalGroups()
logger.info('All physical groups in the model ' + repr(data)
            + ' Names : ' + repr([model.getPhysicalName(*dimtag) for dimtag in data]))

logger.info('Done generating the gmsh geometrical model')
gmsh.write("%s.brep"%name)
os.system("gmsh %s.brep &" %name)

logger.info('Start defining a mesh refinement constraint')
constr_pts = [pt for ll in pattern_ll for pt in ll.vertices]
fine_pts = [pt for pt in constr_pts if (pt.coord[0] % 1 < p[0]/2. or pt.coord[0] % 1 > 1. - p[0]/2.)]
fine_pts = geo.remove_duplicates(fine_pts)

f = msh.set_mesh_refinement([r, a], [r/2, a/3], attractors={'points':fine_pts}, sigmoid_interpol=True)
msh.set_background_mesh(f)
logger.info('Done defining a mesh refinement constraint')
macro_bndry = macro_ll.sides
micro_bndry = list()
rve_s.get_boundary(recursive=True)
micro_bndry = [geo.gather_boundary_fragments(rve_s.boundary, M_ln) for M_ln in macro_bndry]
dirct = [(M_ln.def_pts[-1].coord - M_ln.def_pts[0].coord) for M_ln in macro_bndry]
logger.debug('value and type of dirct items : ' + repr([(i, type(i)) for i in dirct]))
for  i, crvs in enumerate(micro_bndry):
    msh.order_curves(crvs, dirct[i%2], orientation=True)
logger.debug("length of micro_bndry list : " + str(len(micro_bndry)))

msh.set_periodicity_pairs(micro_bndry[0], micro_bndry[2])
msh.set_periodicity_pairs(micro_bndry[1], micro_bndry[3])

factory.remove([(1, l.tag) for l in macro_ll.sides])
for  l in macro_ll.sides:
        l.tag = None

logger.debug('Mesh.SaveAll option value before change : ' + str(gmsh.option.getNumber('Mesh.SaveAll')))
gmsh.option.setNumber('Mesh.SaveAll',0)
logger.debug('Mesh.SaveAll option value after change : ' + str(gmsh.option.getNumber('Mesh.SaveAll')))
gmsh.model.mesh.generate(2)
gmsh.write("%s.msh"%name)
os.system("gmsh %s.msh &" %name)
gmsh.fltk.run()