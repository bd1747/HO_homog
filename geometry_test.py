# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 11:00:18 2018

@author: Baptiste
"""

import os
import gmsh
import geometry as geo
import math
import numpy as np
import matplotlib.pyplot as plt
import copy

# nice shortcuts
model = gmsh.model
factory = model.occ

SR2 = math.sqrt(2.)

# plt.ion() #* interactive mode
plt.ioff()
geo.init_geo_tools()

print("matplotlib pyplot interactive mode : %s" %(plt.isinteractive()))

def test_Point():
    """
    Test of the Point class.
    """

    name = "test_Point"
    gmsh.model.add(name)
    coords = [(0.,0.), (0.,2.5), (SR2,SR2)]
    coords = [np.array(c) for c in coords]
    pts = [geo.Point(c) for c in coords]

    print("Before calling the API addPoint function :")
    for pt in pts :
        print("Point tag  : {}, coordinates : {}".format(pt.tag, pt.coord))
    for pt in pts:
        pt.add_gmsh()
    print("After calling the API addPoint function :")
    for pt in pts :
        print("Point tag  : {}, coordinates : {}".format(pt.tag, pt.coord))

    factory.synchronize()
    data = gmsh.model.getEntities()
    print("model name : %s"%name)
    print(data)

    fig = plt.figure()
    plt.axis('equal')
    for pt in pts:
        pt.plot()
    plt.pause(0.1)
    gmsh.write("%s.brep"%name)
    os.system("gmsh %s.brep &" %name)

def test_Line():
    """
    Test of the Line class.

    Position : OK
    Mesh : OK
    """

    name = "test_Line"
    gmsh.model.add(name)

    coords = [(0.,0.), (0.,2.5), (SR2, SR2)]
    mesh_size = [0.1, 0.1, 0.01]
    coords = [np.array(c) for c in coords]
    pts = [geo.Point(c, m_s) for c, m_s in zip(coords, mesh_size)]
    lines = [geo.Line(pts[0], pts[1]),geo.Line(pts[0], pts[2])]

    for ln in lines:
        ln.add_gmsh()

    factory.synchronize()
    data = gmsh.model.getEntities()
    print("model name : %s \n "%name, data)
    gmsh.model.mesh.generate(1) #We can generatate 1D meshes for the 2 lines

    fig = plt.figure()
    plt.axis('equal')
    for ln in lines:
        ln.plot()
    plt.pause(0.1)
    gmsh.model.mesh.generate(1)
    gmsh.write("%s.brep"%name)
    gmsh.write("%s.msh"%name)
    os.system("gmsh %s.brep &" %name)
    os.system("gmsh %s.msh &" %name)

def test_Arc():
    """
    Test of the Arc class.
    Position : OK
    Mesh : OK
    """

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

    factory.synchronize()
    data = gmsh.model.getEntities()
    print("model name : %s \n"%name, data)
    gmsh.model.mesh.generate(1) #We can generatate 1D meshes for the 2 lines

    fig, ax = plt.subplots()
    plt.axis('equal')
    for pt in pts:
        pt.plot()
    for ln in lines:
        ln.plot()
    for arc in arcs:
        arc.plot()
    plt.pause(0.1)
    gmsh.model.mesh.generate(1)
    gmsh.write("%s.brep"%name)
    gmsh.write("%s.msh"%name)
    os.system("gmsh %s.brep &" %name)
    os.system("gmsh %s.msh &" %name)

def test_LineLoop():
    """
    Test of the LineLoop class.
    Instantiate (explicit and implicit) : #*OK
    Add to the gmsh geometry model : #*OK
    Plot : #*OK
    #TODO : Tester methodes round_corner
    #TODO : Test symétries
    """

    name = "test_LineLoop"
    gmsh.model.add(name)
    #* Explicit constructor
    coords = [(0.05,0.), (1.8,0.), (2.0, 0.2), (2.0, 1.95), (1.95, 2.0), (0.2, 2.0), (0., 1.8), (0.,0.05)]
    mesh_size = [0.01]*len(coords)
    pts = [geo.Point(np.array(c), m_s) for c, m_s in zip(coords, mesh_size)]
    lines = [geo.Line(pts[0], pts[1]), geo.Line(pts[2], pts[3]), geo.Line(pts[4], pts[5]), geo.Line(pts[6], pts[7])]
    centers = [(1.8,0.2), (1.95, 1.95), (0.2, 1.8), (0.05,0.05)]
    centers = [geo.Point(np.array(c)) for c in centers]
    arcs = [geo.Arc(pts[1], centers[0], pts[2]),geo.Arc(pts[3], centers[1], pts[4]), geo.Arc(pts[5], centers[2], pts[6]), geo.Arc(pts[7], centers[3], pts[0])]

    elmts_1D = [item for pair in zip(lines, arcs) for item in pair]
    ll_1 = geo.LineLoop(elmts_1D, explicit=True)

    #* Implicit constructor
    coords = [(1.,1.), (3., 1.), (3., 3.), (1., 3.)]
    mesh_size = [0.01]*len(coords)
    vertc = [geo.Point(np.array(c), m_s) for c, m_s in zip(coords, mesh_size)]
    ll_2 = geo.LineLoop(vertc, explicit=False)

    ll_1.add_gmsh()
    ll_2.add_gmsh()
    factory.synchronize()
    data = gmsh.model.getEntities()
    print("model name : %s \n"%name, data)
    gmsh.model.mesh.generate(1) #We can generatate 1D meshes for the 2 lines

    fig, ax = plt.subplots()
    plt.axis('equal')
    ll_1.plot()
    ll_2.plot()
    plt.pause(0.1)
    gmsh.model.mesh.generate(1)
    gmsh.write("%s.brep"%name)
    gmsh.write("%s.msh"%name)
    os.system("gmsh %s.brep &" %name)
    os.system("gmsh %s.msh &" %name)

def test_PlaneSurface():
    """
    Test of the PlaneSurface class.
    Geometry of surf_1, surf_2 and surf_with_hole : #*OK
    Mesh of the 3 surfaces : #*OK
    """

    name = "test_PlaneSurface"
    gmsh.model.add(name)

    coords = [(0.,0.05), (0.05,0.), (1.8,0.), (2.0, 0.2), (2.0, 1.95),  (1.95, 2.0), (0.2, 2.0), (0., 1.8)]
    pts = [geo.Point(np.array(c), 0.03) for c in coords]
    lines = [geo.Line(pts[2*i-1], pts[2*i]) for i in range(len(pts)//2)]
    centers = [geo.Point(np.array(c), 0.03) for c in [(0.05,0.05), (1.8,0.2), (1.95, 1.95), (0.2, 1.8)]]
    arcs = [geo.Arc(pts[2*i], centers[i], pts[2*i+1]) for i in range(len(pts)//2)]
    elmts_1D = [item for pair in zip(lines, arcs) for item in pair]
    ll_1 = geo.LineLoop(elmts_1D, explicit=True)
    coords = [(1.,1.), (3., 1.), (3., 3.), (1., 3.)]
    vertc = [geo.Point(np.array(c), 0.01) for c in coords]
    ll_2 = geo.LineLoop(vertc, explicit=False)

    surf_1 = geo.PlaneSurface(ll_1)
    surf_2 = geo.PlaneSurface(ll_2)

    rect_vtcs = [geo.Point(np.array(c), 0.05) for c in [(4,2), (4,4), (6,4), (6,2)]]
    hole_vtcs = [geo.Point(np.array(c), 0.02) for c in [(5-0.1,3), (5,3-0.5), (5+0.1,3), (5,3+0.5)]]
    rect_ll = geo.LineLoop(rect_vtcs, explicit=False)
    hole_ll = geo.LineLoop(hole_vtcs, explicit=False)
    surf_with_hole = geo.PlaneSurface(rect_ll, [hole_ll])

    all_surf = [surf_1, surf_2, surf_with_hole]
    print("Model : %s \n Add PlaneSurface instances to a gmsh model. \n Surfaces tag :"%name)
    for s in all_surf:
        s.add_gmsh()
        print(s.tag)

    factory.synchronize()
    data = gmsh.model.getEntities()
    print("model name : %s"%name)
    print(data)
    print("Option Mesh.SaveAll is set to 1. Initial value : %i" %gmsh.option.getNumber('Mesh.SaveAll'))
    gmsh.option.setNumber('Mesh.SaveAll',1)
    gmsh.model.mesh.generate(2)
    gmsh.write("%s.brep"%name)
    gmsh.write("%s.msh"%name)
    os.system("gmsh %s.brep &" %name)
    os.system("gmsh %s.msh &" %name)


def test_bool_ops():
    """
    Test of boolean operations performed on PlaneSurfaces instances.
    cut : OK
    intersection : geometry : OK
    intersection : mesh size : No. Very coarse mesh if the characteristic length is set from the values at Points
    #? What the second output of the boolean operation functions are?
    """

    name = "test_PlaneSurface_bool_ops"
    gmsh.model.add(name)

    coords = [(0.,0.05), (0.05,0.), (1.8,0.), (2.0, 0.2), (2.0, 1.95), (1.95, 2.0), (0.2, 2.0), (0., 1.8)]
    pts = [geo.Point(np.array(c), 0.03) for c in coords]
    lines = [geo.Line(pts[2*i-1], pts[2*i]) for i in range(len(pts)//2)]
    centers = [geo.Point(np.array(c), 0.03) for c in [(0.05,0.05), (1.8,0.2), (1.95, 1.95), (0.2, 1.8)]]
    arcs = [geo.Arc(pts[2*i], centers[i], pts[2*i+1]) for i in range(len(pts)//2)]
    elmts_1D = [item for pair in zip(lines, arcs) for item in pair]
    ll_1 = geo.LineLoop(elmts_1D, explicit=True)
    coords = [(1.,1.), (3., 1.), (3., 3.), (1., 3.)]
    vertc = [geo.Point(np.array(c), 0.1) for c in coords]
    ll_2 = geo.LineLoop(vertc, explicit=False)
    rect_vtcs = [geo.Point(np.array(c), 0.05) for c in [(4,2), (4,4), (6,4), (6,2)]]
    hole_vtcs = [geo.Point(np.array(c), 0.02) for c in [(5-0.1,3), (5,3-0.5), (5+0.1,3), (5,3+0.5)]]
    rect_ll = geo.LineLoop(rect_vtcs, explicit=False)
    hole_ll = geo.LineLoop(hole_vtcs, explicit=False)

    surf_1 = geo.PlaneSurface(ll_1)
    surf_2 = geo.PlaneSurface(ll_2)
    surf_rect = geo.PlaneSurface(rect_ll)
    surf_hole = geo.PlaneSurface(hole_ll)
    surf_with_hole = surf_rect
    for s in [surf_1, surf_2, surf_rect, surf_hole]:
        s.add_gmsh()
    print("Tags before boolean operations : \n", 
           "surf_1 : %i; surf_2 : %i; surf_rect : %i; surf_hole : %i; surf_with_hole : %i"%(surf_1.tag, surf_2.tag, surf_rect.tag, surf_hole.tag, surf_with_hole.tag))
    surf_with_hole.bool_cut(surf_hole,remove_tools=True)
    surf_1.bool_intersect(surf_2,remove_tools=False)
    factory.synchronize()
    data = gmsh.model.getEntities()
    print("model name : %s"%name)
    print(data)
    gmsh.model.mesh.generate(2)
    gmsh.write("%s.brep"%name)
    gmsh.write("%s.msh"%name)
    os.system("gmsh %s.brep &" %name)
    os.system("gmsh %s.msh &" %name)

def test_ll_modif():
    """
    Test of the three geometry operations that can be applied to LineLoop instances : 
        - offset
        - round_corner_explicit 
        - round_corner_incircle
    geometry (check with plots) : OK, gmsh model : OK
    """

    name = "test_LineLoop_offset_round_corners"
    gmsh.model.add(name)

    t = math.tan(math.pi/6)
    vertcs_lists = []
    vertcs_lists.append([geo.Point(np.array(c), 0.05) for c in [(0, 0), (2, 0),  (2-SR2, 0+SR2), (2, 2), (2-1, 2+t) ,(0, 2)]]) #angles : pi/4, pi/2 and 2*pi/3
    vertcs_lists.append([geo.Point(np.array(c), 0.05) for c in [(3, 0), (5, 0), (5-SR2, 0+SR2), (5, 2), (5-1, 2+t), (3, 2)]])
    vertcs_lists.append([geo.Point(np.array(c), 0.05) for c in [(6, 0), (8, 0), (8-SR2, 0+SR2), (8, 2), (8-1, 2+t), (6, 2)]])

    lls = [geo.LineLoop(vl, explicit=False) for vl in vertcs_lists]
    fig = plt.figure()
    plt.axis('equal')
    for ll in lls:
        ll_copy = copy.deepcopy(ll) #The plot method will creat sides and the 3 next methods have been designed to work well only if sides is empty.
        ll_copy.plot("blue")

    lls[0].offset(0.1)
    lls[1].round_corner_explicit(0.1)
    lls[2].round_corner_incircle(0.1)
    for ll in lls:
        ll.plot("orange")
    surfs = [geo.PlaneSurface(ll) for ll in lls]
    for s in surfs:
        s.add_gmsh()
    factory.synchronize()
    data = gmsh.model.getEntities()
    print("model name : %s"%name)
    print(data)
    gmsh.model.mesh.generate(2)
    gmsh.write("%s.brep"%name)
    gmsh.write("%s.msh"%name)
    os.system("gmsh %s.brep &" %name)
    os.system("gmsh %s.msh &" %name)


if __name__ == '__main__':
    test_Point()
    test_Line()
    test_Arc()
    test_LineLoop()
    test_PlaneSurface()
    test_ll_modif()
    # test_bool_ops()

    #* Bloc de fin
    plt.show() #* Il faut fermer toutes les fenêtres avant de passer à la GUI gmsh. (pertinent en mode non interactive)
    # gmsh.fltk.run() #! A revoir, ça génère des "kernel died" dans Spyder, pas idéal
    # # gmsh.fltk.initialize()
    # # gmsh.fltk.wait(10) #? Essai
    # gmsh.finalize()
    #plt.show() #! Utiliser one per script selon la doc. Seulement à la fin, quand on est en mode non interactive
