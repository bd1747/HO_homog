# coding: utf-8
"""
Created on 12/06/2019
@author: baptiste

"""
import logging

import gmsh

import ho_homog

geo = ho_homog.geometry
mesh_gen = ho_homog.mesh_generate

geo.init_geo_tools()
geo.set_gmsh_option("Mesh.MshFileVersion", 4.1)
fltk = gmsh.fltk


def test_E11only_offset_RVE():
    logger = logging.getLogger("test_E11only_offset_RVE")
    fltk.initialize()
    fltk.update()
    a = 1
    r = a / 2e2
    thickness = a / 1e2
    panto_rve = mesh_gen.pantograph.pantograph_E11only_RVE(a, thickness, r, name="panto")
    lc_ratio = 1 / 2
    lc_min_max = (lc_ratio * r, lc_ratio * a)
    d_min_max = (2 * r, a)
    panto_rve.main_mesh_refinement(d_min_max, lc_min_max, False)
    panto_rve.mesh_generate()
    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()
    gmsh.write(str(panto_rve.mesh_abs_path))


test_E11only_offset_RVE()
fltk.run()