# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:00:18 2018

@author: Baptiste
"""
import gmsh 
import geometry as geo
import math 
import numpy as np
import matplotlib.pyplot as plt

# plt.ion() #* interactive mode
plt.ioff()
geo.init_geo_tools()

print("matplotlib pyplot interactive mode : %s" %(plt.isinteractive()))

#%% #*  Test of Point class
name = "test_Point"
gmsh.model.add(name)
sqrt2 = math.sqrt(2.)
coords = [(0.,0.), (0.,2.5), (sqrt2,sqrt2)]
coords = [np.array(c) for c in coords]
pts = [geo.Point(c) for c in coords]

print("Before calling the API addPoint function :")
for pt in pts :
    print("Point tag  : {}, coordinates : {}, Point in geometry model indicator : {}".format(pt.tag, pt.coord, pt.in_model))

for pt in pts:
    pt.add_gmsh()

print("After calling the API addPoint function :")
for pt in pts :
    print("Point tag  : {}, coordinates : {}, Point in geometry model indicator : {}".format(pt.tag, pt.coord, pt.in_model))

gmsh.model.occ.synchronize()
data = gmsh.model.getEntities()
print("model name : %s"%name)
print(data)
#gmsh.fltk.run()
# gmsh.finalize()

fig = plt.figure()
plt.axis('equal')
for pt in pts:
    pt.plot()
plt.pause(0.1)

#%% #*  Test of Line class
name = "test_Line"
gmsh.model.add(name)

sqrt2 = math.sqrt(2.)
coords = [(0.,0.), (0.,2.5), (sqrt2,sqrt2)]
mesh_size = [0.1, 0.1, 0.01]
coords = [np.array(c) for c in coords]
pts = [geo.Point(c, m_s) for c, m_s in zip(coords, mesh_size)]

lines = [geo.Line(pts[0], pts[1]),geo.Line(pts[0], pts[2])]

for ln in lines:
    ln.add_gmsh()

gmsh.model.occ.synchronize()
data = gmsh.model.getEntities()
print("model name : %s"%name)
print(data)
gmsh.model.mesh.generate(1) #We can generatate 1D meshes for the 2 lines


fig = plt.figure()
plt.axis('equal')
for pt in pts:
    pt.plot()
for ln in lines:
    ln.plot()
plt.pause(0.1)

#* Test Line :
#* positions : OK, mesh : OK

#%% #*  Test of Arc class
name = "test_Arc"
gmsh.model.add(name)

coords = [(0.05,0.), (1.8,0.), (2.0, 0.2), (2.0, 1.95),  (1.95, 2.0), (0.2, 2.0), (0., 1.8), (0.,0.05)]
mesh_size = [0.01]*len(coords)
pts = [geo.Point(np.array(c), m_s) for c, m_s in zip(coords, mesh_size)]
lines = [geo.Line(pts[0], pts[1]), geo.Line(pts[2], pts[3]), geo.Line(pts[4], pts[5]), geo.Line(pts[6], pts[7])]
centers = [(1.8,0.2), (1.95, 1.95), (0.2, 1.8), (0.05,0.05)]
centers = [geo.Point(np.array(c)) for c in centers]
arcs = [geo.Arc(pts[1], centers[0], pts[2]),geo.Arc(pts[3], centers[1], pts[4]), geo.Arc(pts[5], centers[2], pts[6]), geo.Arc(pts[7], centers[3], pts[0])]
for ln in lines:
    ln.add_gmsh()
for arc in arcs:
    arc.add_gmsh()

gmsh.model.occ.synchronize()
data = gmsh.model.getEntities()
print("model name : %s"%name)
print(data)
gmsh.model.mesh.generate(1) #We can generatate 1D meshes for the 2 lines

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.axis('equal')
for pt in pts:
    pt.plot()
for ln in lines:
    ln.plot()
for arc in arcs:
    arc.plot()
plt.pause(0.1)

#* Test Arc class : OK
#* positions : OK, mesh : OK

#%% #* Test LineLoop class
name = "test_LineLoop"
gmsh.model.add(name)
#* Explicit constructor
coords = [(0.05,0.), (1.8,0.), (2.0, 0.2), (2.0, 1.95),  (1.95, 2.0), (0.2, 2.0), (0., 1.8), (0.,0.05)]
mesh_size = [0.01]*len(coords)
pts = [geo.Point(np.array(c), m_s) for c, m_s in zip(coords, mesh_size)]
lines = [geo.Line(pts[0], pts[1]), geo.Line(pts[2], pts[3]), geo.Line(pts[4], pts[5]), geo.Line(pts[6], pts[7])]
centers = [(1.8,0.2), (1.95, 1.95), (0.2, 1.8), (0.05,0.05)]
centers = [geo.Point(np.array(c)) for c in centers]
arcs = [geo.Arc(pts[1], centers[0], pts[2]),geo.Arc(pts[3], centers[1], pts[4]), geo.Arc(pts[5], centers[2], pts[6]), geo.Arc(pts[7], centers[3], pts[0])]
for ln in lines:
    ln.add_gmsh()
for arc in arcs:
    arc.add_gmsh()
elmts_1D = [item for pair in zip(lines, arcs) for item in pair]
ll_1 = geo.LineLoop(elmts_1D, explicit=True)


#* Implicit constructor
coords = [(1.,1.), (3., 1.), (3., 3.), (1., 3.)]
mesh_size = [0.01]*len(coords)
vertc = [geo.Point(np.array(c), m_s) for c, m_s in zip(coords, mesh_size)]
ll_2 = geo.LineLoop(vertc, explicit=False)

ll_1.add_gmsh()
ll_2.add_gmsh()
gmsh.model.occ.synchronize()
data = gmsh.model.getEntities()
print("model name : %s"%name)
print(data)
gmsh.model.mesh.generate(1) #We can generatate 1D meshes for the 2 lines

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.axis('equal')
ll_1.plot()
ll_2.plot()
plt.pause(0.1)

plt.show() #* Il faut fermer toutes les fenêtres avant de passer à la GUI gmsh. (pertinent en mode non interactive)

#gmsh.fltk.run() #! A revoir, ça génère des "kernel died" dans Spyder, pas idéal
#gmsh.finalize()

# plt.show() #! Utiliser one per script selon la doc. Seulement à la fin, quand on est en mode non interactive

#%%
def test(param):
    if param:
        return
    else:
        print(param)