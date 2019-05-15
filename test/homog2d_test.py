# coding: utf-8
"""
Created on 09/01/2019
@author: baptiste

"""

import numpy as np

import meshio
from ho_homog import geometry, homog2d, materials, mesh_generate_2D, part
from pytest import approx


def test_homog_EGG_pantograph_1x1(generate_mesh=False):
    if generate_mesh:
        geometry.init_geo_tools()
        geometry.set_gmsh_option("General.Verbosity", 4)
        geometry.set_gmsh_option('Mesh.Algorithm', 5)
        geometry.set_gmsh_option('Mesh.MshFileVersion', 4.1)
        a = 1
        b, k = a, a/3
        r = 0.02
        panto_test = mesh_generate_2D.Gmsh2DRVE.pantograph(
            a, b, k, r, nb_cells=(1, 1), soft_mat=False, name='panto_rve_1x1')
        lc_ratio = 1/6
        lc_min_max = (lc_ratio*r*a, lc_ratio*a)
        panto_test.main_mesh_refinement((2*r*a, a), lc_min_max, False)

        panto_test.mesh_generate()
        mesh = meshio.read("panto_rve_1x1.msh")
        mesh.points = mesh.points[:, :2]
        geo_only = meshio.Mesh(
            points=mesh.points,
            cells={"triangle": mesh.cells["triangle"]})
        meshio.write("panto_rve_1x1.xdmf", geo_only)

    E, nu = 1., 0.3
    material = materials.Material(E, nu, 'cp')
    gen_vect = np.array([[4., 0.], [0., 8.]])
    rve = part.Fenics2DRVE.file_2_Fenics_2DRVE(
        "panto_rve_1x1.xdmf", gen_vect, material)

    hom_model = homog2d.Fenics2DHomogenization(rve)
    *localzt_dicts, constit_tensors = hom_model.homogenizationScheme('EGG')

    np.set_printoptions(suppress=False, floatmode='fixed', precision=8,
                        linewidth=150)

    Chom_ref = np.array(
        [[2.58608139e-04, 3.45496903e-04, 5.16572422e-12],
         [3.45496903e-04, 3.81860676e-02, 6.48384646e-11],
         [5.16572422e-12, 6.48384646e-11, 3.27924466e-04]])
    D_ref = np.array(
        [[ 3.72630940e-02,  2.20371444e-02,  1.00603288e-09, -1.51425656e-11,  6.18921191e-10, -1.39407898e-03],
         [ 2.20371444e-02, -4.32257286e-03,  3.52427076e-09,  3.08504101e-09,  2.91393310e-09,  1.24155443e-02],
         [ 1.00603288e-09,  3.52427076e-09,  1.30706023e-01,  1.59177545e-02, -9.94987221e-03, -6.34655280e-10],
         [-1.51425656e-11,  3.08504101e-09,  1.59177545e-02,  1.58014087e-01,  1.28902572e-01,  1.06773326e-09],
         [ 6.18921191e-10,  2.91393310e-09, -9.94987221e-03,  1.28902572e-01,  1.16274758e-01,  7.79908277e-10],
         [-1.39407898e-03,  1.24155443e-02, -6.34655280e-10,  1.06773326e-09,  7.79908277e-10,  6.38090435e-05]]
    )

    Chom = constit_tensors['E']['E']
    G = constit_tensors['E']['EGGbis']
    D = (constit_tensors['EG']['EG']
         - np.vstack((G[:, :6], G[:, 6:])) - np.vstack((G[:, :6], G[:, 6:])).T)

    print(Chom)
    print(Chom == approx(Chom_ref))
    print(D)
    print(D == approx(D_ref))
    assert Chom == approx(Chom_ref)
    assert D == approx(D_ref)


test_homog_EGG_pantograph_1x1(False)
