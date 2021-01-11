# coding: utf8
"""
Created on 22/10/2018
@author: baptiste

"""

import os

import gmsh
from ho_homog import geometry as geo
from ho_homog import mesh_tools as msh
import numpy as np
import random
import matplotlib.pyplot as plt

# nice shortcuts
model = gmsh.model
factory = model.occ

plt.ion()

# ? COMPARER AVEC DES MAILLAGES PRODUITS A LA MAIN JUSTE AVEC LA GUI GMSH ?

rect_vtcs = [geo.Point(np.array(c)) for c in [(-2, -1), (2, -1), (2, 1), (-2, 1)]]
mid_N = geo.Point(np.array((0, 1)))
mid_S = geo.Point(np.array((0, -1)))
attrac_NW = geo.Point(np.array((-1, 0.5)))
attrac_mid = geo.Line(mid_N, mid_S)

rect_ll = geo.LineLoop(rect_vtcs, False)
rect_s = geo.PlaneSurface(rect_ll)


def test_MathEval():
    """
    Test of MathEval subclass of the Field base class.

    #* Test OK

    """
    name = "test_MathEval"
    model.add(name)
    rect_vtcs = [geo.Point(np.array(c)) for c in [(-4, -2), (4, -2), (4, 2), (-4, 2)]]
    rect_ll = geo.LineLoop(rect_vtcs, False)
    rect_s = geo.PlaneSurface(rect_ll)
    rect_s.add_gmsh()

    f = msh.MathEvalField("(Cos(3.14*x) * Sin(3.14*y)+1)*0.1+0.005")
    f.add_gmsh()  # Optional
    msh.set_background_mesh(f)

    factory.synchronize()
    model.mesh.generate(2)
    gmsh.write("%s.msh" % name)
    os.system("gmsh %s.msh &" % name)


def test_Min():
    """
    Test of Min subclass of the Field base class
    and of the use of set_background_mesh function with a list of fields as input.
    
    #* Test OK

    """
    name = "test_Min"
    model.add(name)
    rect_vtcs = [geo.Point(np.array(c)) for c in [(-4, -2), (4, -2), (4, 2), (-4, 2)]]
    rect_ll = geo.LineLoop(rect_vtcs, False)
    rect_s = geo.PlaneSurface(rect_ll)
    rect_s.add_gmsh()

    h1 = msh.MathEvalField("0.1")
    h2 = msh.MathEvalField("(x*x)+0.02")
    h1.add_gmsh()
    h2.add_gmsh()
    msh.set_background_mesh([h1, h2])

    factory.synchronize()
    model.mesh.generate(2)
    gmsh.write("%s.msh" % name)
    os.system("gmsh %s.msh &" % name)


def test_Threshold():
    """
    Test of the AttractorField and ThresholdField subclass of the Field base class.
    #* Test OK

    """
    name = "test_Threshold"
    model.add(name)
    rect_vtcs = [geo.Point(np.array(c)) for c in [(-4, -2), (4, -2), (4, 2), (-4, 2)]]
    mid_N = geo.Point(np.array((0, 2)))
    mid_S = geo.Point(np.array((0, -2)))
    attrac_NW = geo.Point(np.array((-2, 1)))
    attrac_mid = geo.Line(mid_N, mid_S)
    rect_ll = geo.LineLoop(rect_vtcs, False)
    rect_s = geo.PlaneSurface(rect_ll)
    rect_s.add_gmsh()

    f = msh.AttractorField(
        points=[attrac_NW], curves=[attrac_mid], nb_pts_discretization=10
    )
    g = msh.ThresholdField(f, 0.2, 1, 0.03, 0.2, True)

    msh.set_background_mesh(g)

    factory.synchronize()
    model.mesh.generate(2)
    gmsh.write("%s.msh" % name)
    os.system("gmsh %s.msh &" % name)


def test_Restrict():
    """
    Test of the Restrict subclass of the Field base class.
    #* Test OK!

    """
    name = "test_Restrict"
    model.add(name)
    vtcs_1 = [geo.Point(np.array(c)) for c in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    vtcs_2 = [geo.Point(np.array(c)) for c in [(0, 0), (2, 0), (2, 2), (0, 2)]]
    ll_1 = geo.LineLoop(vtcs_1, explicit=False)
    ll_2 = geo.LineLoop(vtcs_2, explicit=False)
    surf_1 = geo.PlaneSurface(ll_1)
    surf_2 = geo.PlaneSurface(ll_2)
    surf_1.add_gmsh()
    surf_2.add_gmsh()

    f = msh.MathEvalField("0.1")
    g = msh.MathEvalField("(x-0.25)*(x-0.25)+0.02")
    h = msh.RestrictField(g, surfaces=[surf_1])
    msh.set_background_mesh([f, h])

    factory.synchronize()
    model.mesh.generate(2)
    gmsh.write("%s.msh" % name)
    os.system("gmsh %s.msh &" % name)


def test_refine_function():
    """
    Test of the function that has been designed to quickly specify a refinement constraint.
    #* Test OK

    """
    name = "test_refine_function"
    model.add(name)
    rect_vtcs = [geo.Point(np.array(c)) for c in [(-4, -2), (4, -2), (4, 2), (-4, 2)]]
    mid_N = geo.Point(np.array((0, 2)))
    mid_S = geo.Point(np.array((0, -2)))
    attrac_NW = geo.Point(np.array((-2, 1)))
    attrac_mid = geo.Line(mid_N, mid_S)
    rect_ll = geo.LineLoop(rect_vtcs, False)
    rect_s = geo.PlaneSurface(rect_ll)
    rect_s.add_gmsh()

    f = msh.set_mesh_refinement(
        [0.2, 1],
        [0.03, 0.2],
        attractors={"points": [attrac_NW], "curves": [attrac_mid]},
        sigmoid_interpol=True,
    )
    msh.set_background_mesh(f)

    factory.synchronize()
    model.mesh.generate(2)
    gmsh.write("%s.msh" % name)
    os.system("gmsh %s.msh &" % name)


def test_fctn_restrict():
    """
    Test of the function that has been designed to quickly specify a refinement constraint with a restriction to a given surface.
    #* Test OK

    """
    name = "test_Restrict_with_fctn"
    model.add(name)
    vtcs_1 = [geo.Point(np.array(c)) for c in [(-1, -1), (1, -1), (1, 1), (-1, 1)]]
    vtcs_2 = [geo.Point(np.array(c)) for c in [(0, 0), (2, 0), (2, 2), (0, 2)]]
    ll_1 = geo.LineLoop(vtcs_1, explicit=False)
    ll_2 = geo.LineLoop(vtcs_2, explicit=False)
    surf_1 = geo.PlaneSurface(ll_1)
    surf_2 = geo.PlaneSurface(ll_2)
    surf_1.add_gmsh()
    surf_2.add_gmsh()

    f = msh.MathEvalField("0.1")
    g = msh.MathEvalField("(x-0.25)*(x-0.25)+0.02")
    h = msh.MathEvalField("(y-0.25)*(y-0.25)+0.02")
    gr = msh.RestrictField(g, surfaces=[surf_1], curves=surf_1.ext_contour.sides)
    hr = msh.RestrictField(
        h, surfaces=[surf_2], curves=surf_2.ext_contour.sides + surf_1.ext_contour.sides
    )
    msh.set_background_mesh([f, gr, hr])

    factory.synchronize()
    model.mesh.generate(2)
    gmsh.write("%s.msh" % name)
    os.system("gmsh %s.msh &" % name)


# ?               |￣￣￣￣￣￣￣￣￣|
# ?               | All field tests |
# ?               | succeeded !     |
# ?               | ＿＿＿＿＿＿＿＿_|
# ?        (\__/) ||
# ?        (•ㅅ•) ||
# ?        / 　 づ


def test_translation2matrix():
    mtx1 = msh.translation2matrix(np.array([5, 0]))
    mtx2 = msh.translation2matrix(np.array([1, 2, 3]))
    mtx3 = msh.translation2matrix(np.array([1, 1, 0]), 1)
    print("translation by the vector (5,0) in the XY plane :\n", mtx1)
    print("translation by the vector (1, 2, 3):\n", mtx2)
    print(
        "translation in the direction of the first bissectrice of the XY plane by a distance of 1:\n",
        mtx3,
    )


def test_periodic():
    name = "periodic"
    model.add(name)

    vtcs = [
        geo.Point(np.array(c), 0.5)
        for c in [(-2, +1), (0, +1), (+2, +1), (+2, -1), (0, -1), (-2, -1)]
    ]
    vtcs[0].mesh_size = 0.01
    vtcs[2].mesh_size = 0.05
    sides = [geo.Line(vtcs[i - 1], vtcs[i]) for i in range(len(vtcs))]
    surf = geo.PlaneSurface(geo.LineLoop(sides, explicit=True))
    surf.add_gmsh()
    data = gmsh.model.getEntities()
    print("model name : %s" % name)
    print(data)
    factory.synchronize()
    msh.set_periodicity_pairs([sides[3]], [sides[0]], np.array((4, 0)))
    msh.set_periodicity_pairs(
        [sides[-1], sides[-2]], [sides[1], sides[2]], np.array((0, -2))
    )
    gmsh.model.mesh.generate(2)
    gmsh.write("%s.brep" % name)
    gmsh.write("%s.msh" % name)
    os.system("gmsh %s.brep &" % name)
    os.system("gmsh %s.msh &" % name)


def test_order_curves():
    """
    Test of the order_curves function.
    Shoud be run after the test of the gather_boundary_fragments function.

    Test of the ordering : OK
    Reverse functionnality : #! not tested yet
    """

    name = "test_order"
    gmsh.model.add(name)
    rect_vtcs = [(-4, 2), (-4, -2), (4, -2), (4, 2)]
    rect_vtcs = [geo.Point(np.array(c), 0.2) for c in rect_vtcs]
    line_N = geo.Line(geo.Point(np.array((-4, 2))), geo.Point(np.array((4, 2))))
    line_W = geo.Line(geo.Point(np.array((-4, -2))), geo.Point(np.array((-4, 2))))
    line_S = geo.Line(geo.Point(np.array((-4, -2))), geo.Point(np.array((4, -2))))
    line_E = geo.Line(geo.Point(np.array((4, -2))), geo.Point(np.array((4, 2))))
    lines = {"N": line_N, "E": line_E, "S": line_S, "W": line_W}
    holes = list()
    hole_shape = [np.array(c) for c in [(-0.2, 0), (0, -0.4), (0.2, 0), (0, 0.4)]]
    translations = [
        (-3, 2.1),
        (-1, 2.1),
        (1, 2.1),
        (3, 2.1),
        (-3, -2.1),
        (-1, -2.1),
        (1, -2.1),
        (3, -2.1),
        (4.1, -1),
        (4.1, 1),
        (-4.1, -1),
        (-4.1, 1),
    ]
    translations = [np.array(t) for t in translations]
    for t in translations:
        holes.append([geo.Point(c + t, 0.05) for c in hole_shape])

    rect_ll = geo.LineLoop(rect_vtcs, explicit=False)
    hole_ll = [geo.LineLoop(h, explicit=False) for h in holes]
    rect_s = geo.PlaneSurface(rect_ll)
    hole_s = [geo.PlaneSurface(ll) for ll in hole_ll]
    final_s = geo.bool_cut_S(rect_s, hole_s, remove_body=False, remove_tool=False)
    factory.synchronize()
    final_s = final_s[0]
    final_s.get_boundary(recursive=True)
    fig, ax = plt.subplots()
    ax.set_xlim(-4.1, 4.1)
    ax.set_ylim(-2.1, 2.1)
    for crv in final_s.boundary:
        crv.plot("blue")
    boundaries = {"N": [], "E": [], "S": [], "W": []}
    for key, line in lines.items():
        boundaries[key] = geo.gather_boundary_fragments(final_s.boundary, line)
        random.shuffle(boundaries[key])
    fig, ax = plt.subplots()
    ax.set_xlim(-4.1, 4.1)
    ax.set_ylim(-2.1, 2.1)
    colors = {"N": "red", "E": "green", "S": "orange", "W": "blue"}
    for key in boundaries.keys():
        for l in boundaries[key]:
            l.plot(colors[key])
            plt.pause(0.5)  # In order to see the order of the curves
    basis = np.array(((1.0, 0.0), (0.0, 1.0), (0.0, 0.0)))
    dir_v = {"N": basis[:, 0], "E": basis[:, 1], "S": basis[:, 0], "W": basis[:, 1]}
    for key, lns in boundaries.items():
        msh.order_curves(lns, dir_v[key])
    fig, ax = plt.subplots()
    ax.set_xlim(-4.1, 4.1)
    ax.set_ylim(-2.1, 2.1)
    for key in boundaries.keys():
        for l in boundaries[key]:
            l.plot(colors[key])
            plt.pause(0.5)  # In order to see the order of the curves
    factory.synchronize()
    plt.show(block=True)


if __name__ == "__main__":
    geo.init_geo_tools()
    # test_MathEval()
    # test_Min()
    # test_Threshold()
    test_refine_function()
    # test_Restrict()
    # test_fctn_restrict()

    # test_translation2matrix()
    # test_periodic()
    # test_order_curves()
