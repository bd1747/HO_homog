# coding: utf-8

import gmsh
import ho_homog.geometry as geo
import ho_homog.mesh_generate as mg

geo.init_geo_tools()
geo.set_gmsh_option("Mesh.MshFileVersion", 4.1)


def test_kagome():
    """ Generate a mesh that corresponds to one unit cell of 'kagome' microstructure.
    Two cases : alpha = 0.5, fully open, alpha = 0.1 triangles almost close"""
    a = 1
    junction_thinness = 1e-2
    for alpha in [0.1, 0.5]:
        rve = mg.kagome.kagome_RVE(
            0.5, junction_thinness, a=a, name=f"kagome_{alpha:.2f}"
        )
        lc_ratio = 1 / 2
        lc_min_max = (lc_ratio * junction_thinness * a, lc_ratio * a)
        d_min_max = (2 * junction_thinness * a, a)
        rve.main_mesh_refinement(d_min_max, lc_min_max, False)
        rve.mesh_generate()
        gmsh.model.mesh.renumberNodes()
        gmsh.model.mesh.renumberElements()
        gmsh.write(str(rve.mesh_abs_path))
