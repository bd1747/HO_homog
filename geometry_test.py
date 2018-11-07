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
    hole_vtcs_1 = [geo.Point(np.array(c), 0.02) for c in [(5-0.1,3), (5,3-0.5), (5+0.1,3), (5,3+0.5)]]
    hole_vtcs_2 = [geo.Point(np.array(c), 0.02) for c in [(5, 3-0.1), (5-0.5, 3), (5, 3+0.1), (5+0.5, 3)]]
    rect_ll = geo.LineLoop(rect_vtcs, explicit=False)
    hole_ll_1= geo.LineLoop(hole_vtcs_1, explicit=False)
    hole_ll_2= geo.LineLoop(hole_vtcs_2, explicit=False)

    surf_1 = geo.PlaneSurface(ll_1)
    surf_2 = geo.PlaneSurface(ll_2)
    surf_rect = geo.PlaneSurface(rect_ll)
    surf_hole_1 = geo.PlaneSurface(hole_ll_1)
    surf_hole_2 = geo.PlaneSurface(hole_ll_2)
    for s in [surf_1, surf_2, surf_rect, surf_hole_1, surf_hole_2]:
        s.add_gmsh()
    print("Tags before boolean operations : \n", 
           "surf_1 : %i; surf_2 : %i; surf_rect : %i; surf_hole_1 : %i; surf_hole_2 : %i"%(surf_1.tag, surf_2.tag, surf_rect.tag, surf_hole_1.tag, surf_hole_2.tag))
    surf_with_hole = geo.bool_cut_S(surf_rect, [surf_hole_1, surf_hole_2], remove_tool=True)
    print(surf_with_hole)
    surf_with_hole = surf_with_hole[0]
    surf_inter = geo.bool_intersect_S(surf_1, surf_2, False, False)
    print(surf_inter)
    surf_inter = surf_inter[0]
    factory.synchronize()
    data = gmsh.model.getEntities()
    print("model name : %s"%name)
    print(data)
    print("Tags after boolean operations : \n", 
           "surf_1 : %s; surf_2 : %s; surf_rect : %s; surf_hole_1 : %s; surf_hole_2 : %s, surf_with_hole : %s; surf_inter:%s "%(surf_1.tag, surf_2.tag, surf_rect.tag, surf_hole_1.tag, surf_hole_2.tag, surf_with_hole.tag, surf_inter.tag))
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

def test_gather():
    """
    Test of the gather_boundary_fragments function.
    """

    name = "test_gather"
    gmsh.model.add(name)
    rect_vtcs = [(-4, 2), (-4, -2), (4, -2), (4, 2)]
    rect_vtcs = [geo.Point(np.array(c), 0.2) for c in rect_vtcs]
    line_N = geo.Line(geo.Point(np.array((-4, 2))), geo.Point(np.array((4, 2))))
    line_W = geo.Line(geo.Point(np.array((-4, -2))), geo.Point(np.array((-4, 2))))
    line_S = geo.Line(geo.Point(np.array((-4, -2))), geo.Point(np.array((4, -2))))
    line_E = geo.Line(geo.Point(np.array((4, -2))), geo.Point(np.array((4, 2))))
    holes = list()
    hole_shape = [np.array(c) for c in [(-0.2, 0), (0, -0.4), (0.2, 0), (0, 0.4)]]
    translations = [(-3, 2.1), (-1, 2.1), (1, 2.1), (3, 2.1),
                    (-3, -2.1), (-1, -2.1), (1, -2.1), (3, -2.1),
                    (4.1, -1), (4.1, 1),
                    (-4.1, -1), (-4.1, 1)]
    translations = [np.array(t) for t in translations]
    for t in translations:
        holes.append([geo.Point(c+t, 0.05) for c in hole_shape])

    rect_ll = geo.LineLoop(rect_vtcs, explicit=False)
    hole_ll = [geo.LineLoop(h, explicit=False) for h in holes]
    rect_s = geo.PlaneSurface(rect_ll)
    hole_s = [geo.PlaneSurface(ll) for ll in hole_ll]
    #final_s = geo.bool_cut_S(rect_s, hole_s, remove_body=True, remove_tool=True) #!ERREUR CAR LA SUPPRESSION DES ELEMENTS BOSCULE LE SYSTEME DE TAG
    final_s = geo.bool_cut_S(rect_s, hole_s, remove_body=False, remove_tool=False)
    factory.synchronize()
    print("length of final_s : ", len(final_s))
    final_s = final_s[0]
    final_s.get_boundary(recursive=True)

    plt.figure()
    plt.axis('equal')
    for crv in final_s.boundary:
        # print(c.__dict__,type(c))
        crv.plot("blue")
    boundary_N = geo.gather_boundary_fragments(final_s.boundary, line_N)
    boundary_W = geo.gather_boundary_fragments(final_s.boundary, line_W)
    boundary_S = geo.gather_boundary_fragments(final_s.boundary, line_S)
    boundary_E = geo.gather_boundary_fragments(final_s.boundary, line_E)
    # print(boundary_N)
    # print(boundary_W)
    plt.figure()
    plt.axis('equal')
    colors = ['red', 'green', 'orange', 'blue']
    for l_list,c in zip([boundary_N, boundary_W, boundary_S, boundary_E], colors):
        for l in l_list:
            l.plot(c)
    factory.synchronize()

    gmsh.model.mesh.generate(2)
    gmsh.write("%s.brep"%name)
    gmsh.write("%s.msh"%name)
    os.system("gmsh %s.brep &" %name)
    os.system("gmsh %s.msh &" %name)


def test_remove_ll_duplicates():
    """
    Test of the remove_duplicates static method of the class LineLoop.
    Results : OK
    """
    pts_1 = [(2, 1), (-2, 1), (-2, -1), (2, -1)]
    pts_1 = [geo.Point(np.array(c)) for c in pts_1]
    pts_2 = [(-2, -1), (2, -1), (2, 1), (-2, 1)]
    pts_2 = [geo.Point(np.array(c)) for c in pts_2]
    lines_1 = [[(3, 4),(-3, 4)], [(-3, 4), (-3, -4)], [(-3, -4), (3, -4)], [(3, -4), (3, 4)]]
    lines_2 = [[(-3, 4), (-3, -4)], [(-3, -4), (3, -4)], [(3, -4), (3, 4)], [(3, 4),(-3, 4)]]
    lines_1 = [geo.Line(*[geo.Point(np.array(c)) for c in cc]) for cc in lines_1]
    lines_2 = [geo.Line(*[geo.Point(np.array(c)) for c in cc]) for cc in lines_2]
    ll_pts_1 = geo.LineLoop(pts_1, explicit=False)
    ll_pts_2 = geo.LineLoop(pts_2, explicit=False)
    ll_lines_1 = geo.LineLoop(lines_1, explicit=True)
    ll_lines_2 = geo.LineLoop(lines_2, explicit=True)
    all_ll = [ll_lines_2, ll_pts_1, ll_pts_2, ll_lines_1]
    print(all_ll)
    plt.figure()
    plt.axis('equal')
    colors = ['red', 'green', 'orange', 'blue']
    for ll, c in zip(all_ll, colors):
        ll_copy = copy.deepcopy(ll) #The plot method will creat sides and the 3 next methods have been designed to work well only if sides is empty.
        ll_copy.plot(c)
        plt.pause(0.2)
    unique_ll = geo.LineLoop.remove_duplicates(all_ll)
    print(unique_ll)
    plt.figure()
    plt.axis('equal')
    for ll, c in zip(unique_ll[0], colors):
        ll.plot(c)
        plt.pause(0.2)

def test_physical_group():
    """
    Test of the PhysicalGroup class.
    Group membership in the gmsh model : OK
    Colors in the gmsh GUI opened with gmsh.fltk.run() : OK
    #TODO : Vérifier l'export des physical groups dans le maillage
    """

    name = "test_PhysicalGroup"
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

    surf_group = geo.PhysicalGroup([surf_1, surf_with_hole], 2)
    line_group = geo.PhysicalGroup(ll_2.sides, 1)
    surf_group.add_gmsh()
    # surf_group.add_to_group(surf_2) #* Raise an AttributeError, OK
    line_group.add_to_group(ll_1.sides)
    line_group.add_gmsh()
    surf_group.set_color([255, 65, 0, 255])
    line_group.set_color([120, 246, 0, 255])


    gmsh.model.mesh.generate(2)
    gmsh.write("%s.brep"%name)
    gmsh.write("%s.msh"%name)
    os.system("gmsh %s.brep &" %name)
    os.system("gmsh %s.msh &" %name)

    gmsh.fltk.run()

if __name__ == '__main__':
    # test_Point()
    # test_Line()
    # test_Arc()
    # test_LineLoop()
    # test_PlaneSurface()
    # test_ll_modif()
    #test_bool_ops()
    # test_gather()
    # test_remove_ll_duplicates()
    test_physical_group()
    
    #* Bloc de fin
    plt.show() #* Il faut fermer toutes les fenêtres avant de passer à la GUI gmsh. (pertinent en mode non interactive)
    # gmsh.fltk.run() #! A revoir, ça génère des "kernel died" dans Spyder, pas idéal
    # # gmsh.fltk.initialize()
    # # gmsh.fltk.wait(10) #? Essai
    # gmsh.finalize()
    #plt.show() #! Utiliser one per script selon la doc. Seulement à la fin, quand on est en mode non interactive
