# coding: utf-8
"""
Created on 12/06/2019
@author: baptiste

"""
from ho_homog import geometry as geo
from ho_homog import mesh_generate_2D as mesh2D
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
    mesh2D.msh_conversion(panto_rve.mesh_abs_path, ".xdmf")
    mesh2D.msh_conversion(panto_part.mesh_abs_path, ".xdmf")
