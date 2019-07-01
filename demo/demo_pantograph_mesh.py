# coding: utf8
"""
Created on 07/11/2018
@author: baptiste

"""

import ho_homog.geometry as geo
import ho_homog.mesh_tools as msh
import numpy as np
import logging
import gmsh
import os
from pathlib import Path
from subprocess import run

# nice shortcuts
model = gmsh.model
factory = model.occ

# * Logging
logger = logging.getLogger(__name__)  # http://sametmax.com/ecrire-des-logs-en-python/
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s", "%H:%M")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

geo.init_geo_tools()
name = "pantograph"
model.add(name)

root = Path(__file__).resolve().parent.joinpath(name)
geo_file = root.with_suffix(".brep")
mesh_file = root.with_suffix(".msh")

os.path.realpath(__file__)


logger.info("Start defining the pantograph geometry")
a = 1
b = 1
k = 0.3
r = a / 20

Lx = 4 * a
Ly = 6 * a + 2 * b
gen_vect = np.array(((Lx, 0.0), (0.0, Ly)))

e1 = np.array((a, 0.0, 0.0))
e2 = np.array((0.0, a, 0.0))
p = np.array((k, 0.0, 0.0))
b = b / a * e2

E1 = geo.Point(e1)
E2 = geo.Point(e2)
E1m = geo.Point(-1 * e1)
E2m = geo.Point(-1 * e2)
O_pt = np.zeros((3,))
L = geo.Point(2 * (e1 + e2))
Lm = geo.Point(2 * (e1 - e2))
M = geo.Point(e1 + 1.5 * e2 + b / 2)
N = geo.Point(2 * (e1 + 1.5 * e2 + b / 2))

contours = list()
contours.append([E1, E2, E1m, E2m])
contours.append([E1, Lm, geo.Point(3 * e1), L])
contours.append(
    [
        E2,
        L,
        geo.translation(L, b / 2 - p),
        geo.translation(L, b),
        geo.translation(E2, b),
        geo.translation(E2, b / 2 + p),
    ]
)
pattern_ll = [geo.LineLoop(pt_list, explicit=False) for pt_list in contours]

pattern_ll += [geo.point_reflection(ll, M) for ll in pattern_ll]
sym_ll = [geo.plane_reflection(ll, N, e1) for ll in pattern_ll]
for ll in sym_ll:
    ll.reverse()
pattern_ll += sym_ll
sym_ll = [geo.plane_reflection(ll, N, e2) for ll in pattern_ll]
for ll in sym_ll:
    ll.reverse()
pattern_ll += sym_ll
pattern_ll = geo.remove_duplicates(pattern_ll)
logger.info("Removing of the line-loops duplicates : Done")
logger.info(f"pattern_ll length : {len(pattern_ll)}")
# * 13 no-redundant LineLoops to define the pantograph microstructure geometry in 1 cell.


for ll in pattern_ll:
    ll.round_corner_incircle(r)
logger.info("Rounding all corners of pattern line-loops : Done")

macro_vtcs = [O_pt, gen_vect[0], gen_vect[0] + gen_vect[1], gen_vect[1]]
macro_ll = geo.LineLoop([geo.Point(c) for c in macro_vtcs])
macro_s = geo.PlaneSurface(macro_ll)

logger.info("Start boolean operations on surfaces")
pattern_s = [geo.PlaneSurface(ll) for ll in pattern_ll]
rve_s = geo.surface_bool_cut(macro_s, pattern_s)
rve_s = rve_s[0]
logger.info("Boolean operations on surfaces : Done")
rve_s_phy = geo.PhysicalGroup([rve_s], 2, "partition_plein")
factory.synchronize()

rve_s_phy.add_gmsh()
factory.synchronize()
data = model.getPhysicalGroups()
logger.info(f"All physical groups in the model : \n {data}")
names = [model.getPhysicalName(*dimtag) for dimtag in data]
logger.info(f"Physical group names: \n {names}")

logger.info("Generate geometry model : Done")
gmsh.write(str(geo_file))
run(f"gmsh {str(geo_file)} &", shell=True, check=True)

logger.info("Start defining a mesh refinement constraint")
constr_pts = [pt for ll in pattern_ll for pt in ll.vertices]
fine_pts = [
    pt
    for pt in constr_pts
    if (pt.coord[0] % 1 < p[0] / 2.0 or pt.coord[0] % 1 > 1.0 - p[0] / 2.0)
]
fine_pts = geo.remove_duplicates(fine_pts)
for pt in fine_pts:
    pt.add_gmsh()
factory.synchronize()
f = msh.set_mesh_refinement(
    (r, a), (r / 4, a / 3), attractors={"points": fine_pts}, sigmoid_interpol=True
)
msh.set_background_mesh(f)
logger.info("Mesh refinement constraint Done")

logger.info("Start defining a periodicity constraint for the mesh")
macro_bndry = macro_ll.sides
rve_s.get_boundary(recursive=True)
micro_bndry = [geo.macro_line_fragments(rve_s.boundary, M_ln) for M_ln in macro_bndry]
dirct = [(M_ln.def_pts[-1].coord - M_ln.def_pts[0].coord) for M_ln in macro_bndry]
logger.debug("value and type of dirct items : " + repr([(i, type(i)) for i in dirct]))
for i, crvs in enumerate(micro_bndry):
    msh.order_curves(crvs, dirct[i % 2], orientation=True)
logger.debug("length of micro_bndry list : " + str(len(micro_bndry)))

msh.set_periodicity_pairs(micro_bndry[0], micro_bndry[2])
msh.set_periodicity_pairs(micro_bndry[1], micro_bndry[3])
logger.info("Periodicity constraint : Done")

logger.info("Cleaning model")
factory.remove([(1, l.tag) for l in macro_ll.sides])
factory.synchronize()
factory.removeAllDuplicates()
factory.synchronize()

geo.set_gmsh_option("Mesh.CharacteristicLengthExtendFromBoundary", 0)
geo.set_gmsh_option("Mesh.SaveAll", 0)
geo.PhysicalGroup.set_group_mesh(1)
gmsh.model.mesh.generate(2)
gmsh.write(str(mesh_file))
run(f"gmsh {str(mesh_file)} &", shell=True, check=True)
gmsh.fltk.run()
