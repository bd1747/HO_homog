# coding: utf-8
"""
Created on 12/06/2019
@author: baptiste

"""
from ho_homog import geometry as geo
from ho_homog import mesh_generate as mesh2D
from ho_homog.toolbox_gmsh import msh_conversion
import gmsh

geo.init_geo_tools()


def test_rve_2_part():
    fltk = gmsh.fltk
    geo.set_gmsh_option("Mesh.MshFileVersion", 4.1)
    fltk.initialize()
    fltk.update()
    a = 1
    b, k = a, a / 3
    r = a / 1e2
    panto_rve = mesh2D.Gmsh2DRVE.pantograph(a, b, k, r, name="couple_panto")
    lc_ratio = 1 / 2
    lc_min_max = (lc_ratio * r, lc_ratio * a)
    d_min_max = (5 * r, a)
    panto_rve.main_mesh_refinement(d_min_max, lc_min_max, False)
    panto_rve.mesh_generate()
    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()
    gmsh.write(str(panto_rve.mesh_abs_path))
    panto_part = mesh2D.Gmsh2DPartFromRVE(panto_rve, (2, 3))
    msh_conversion(panto_rve.mesh_abs_path, ".xdmf")
    msh_conversion(panto_part.mesh_abs_path, ".xdmf")


def test_pantograph_offset():
    geo.set_gmsh_option('Mesh.MshFileVersion', 4.1)
    a = 1
    b, k = a, a/3
    t = 0.02
    panto_test = mesh2D.pantograph_offset_RVE(
        a, b, k, t, nb_cells=(1, 1), soft_mat=False, name='panto_rve_1x1')
    lc_ratio = 1/6
    lc_min_max = (lc_ratio*t*a, lc_ratio*a)
    panto_test.main_mesh_refinement((2*t*a, a), lc_min_max, False)
    panto_test.mesh_generate()
    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()
    gmsh.write("panto_rve_offset.msh")


test_pantograph_offset()
