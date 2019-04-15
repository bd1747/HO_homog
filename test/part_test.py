# coding: utf-8
"""
Created on 09/01/2019
@author: baptiste

"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from subprocess import run

import dolfin as fe
import gmsh
import numpy as np

import ho_homog
import mshr
from pytest import approx

geo = ho_homog.geometry
part = ho_homog.part
mesh_2D = ho_homog.mesh_generate_2D
mat = ho_homog.materials

model = gmsh.model
factory = model.occ
logger = logging.getLogger(__name__) #http://sametmax.com/ecrire-des-logs-en-python/
logger.setLevel(logging.INFO)
if __name__ == "__main__":
    logger_root = logging.getLogger()
    logger_root.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s') # Afficher le temps à chaque message
    file_handler = RotatingFileHandler(f'activity_{__name__}.log', 'a', 1000000)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger_root.addHandler(file_handler) #Pour écriture d'un fichier log
    formatter = logging.Formatter('%(levelname)s :: %(message)s')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger_root.addHandler(stream_handler)

ho_homog.geometry.init_geo_tools()

#? Test du constructeur gmsh_2_Fenics_2DRVE
# a = 1
# b, k = a, a/3
# panto_test = prt.Gmsh2DRVE.pantograph(a, b, k, 0.1, nb_cells=(2, 3), soft_mat=False, name='panto_test')
# panto_test.main_mesh_refinement((0.1,0.5),(0.03,0.3),False)

# panto_test.mesh_generate()
# run(f"gmsh {panto_test.name}.msh &",shell=True, check=True)

# E1, nu1 = 1., 0.3
# E2, nu2 = E1/100., nu1
# E_nu_tuples = [(E1, nu1), (E2, nu2)]
# phy_subdomains = panto_test.phy_surf
# material_dict = dict()
# for coeff, subdo in zip(E_nu_tuples, phy_subdomains):
#     material_dict[subdo.tag] = mat.Material(coeff[0], coeff[1], 'cp')
# rve = prt.Fenics2DRVE.gmsh_2_Fenics_2DRVE(panto_test, material_dict)
#* Test ok

#? Problème de maillage lorsqu'un domaine mou existe
# a = 1
# b, k, r = a, a/3, a/10
# panto_test = prt.Gmsh2DRVE.pantograph(a, b, k, r, soft_mat=True, name='panto_test_soft')
# run(f"gmsh {panto_test.name}.brep &", shell=True, check=True)
# panto_test.main_mesh_refinement((r, 5*r), (r/2, a/3), False)
# panto_test.soft_mesh_refinement((r, 5*r), (r/2, a/3), False)
# panto_test.mesh_generate() #!Sans l'opération de removeDuplicateNodes() dans mesh_generate(). Avec domaine mou obtenu par intersection.
# gmsh.model.mesh.removeDuplicateNodes()
# gmsh.write(f"{panto_test.name}_clean.msh")
# run(f"gmsh {panto_test.name}.msh &", shell=True, check=True)
# run(f"gmsh {panto_test.name}_clean.msh &", shell=True, check=True)
# a = 1
# b, k, r = a, a/3, a/10
# panto_test = prt.Gmsh2DRVE.pantograph(a, b, k, r, soft_mat=True, name='panto_test_soft')
# run(f"gmsh {panto_test.name}.brep &", shell=True, check=True)
# panto_test.main_mesh_refinement((r, 5*r), (r/2, a/3), False)
# panto_test.soft_mesh_refinement((r, 5*r), (2*r, a/3), False)
# panto_test.mesh_generate() #!Sans l'opération de removeDuplicateNodes() dans mesh_generate(). Avec domaine mou obtenu par intersection.
# gmsh.write(f"{panto_test.name}_clean.msh")
# run(f"gmsh {panto_test.name}.msh &", shell=True, check=True)
# run(f"gmsh {panto_test.name}_clean.msh &", shell=True, check=True)
#? Fin de la résolution


#? Test d'un mesh avec un domaine mou, gmsh2DRVE puis import dans FEniCS
# a = 1
# b, k = a, a/3
# panto_test = mesh_generate_2D.Gmsh2DRVE.pantograph(a, b, k, 0.1, nb_cells=(2, 3), soft_mat=True, name='panto_test_mou')
# panto_test.main_mesh_refinement((0.1,0.5),(0.03,0.3),False)
# panto_test.soft_mesh_refinement((0.1,0.5),(0.03,0.3),False)
# panto_test.mesh_generate()
# run(f"gmsh {panto_test.name}.msh &",shell=True, check=True)
# E1, nu1 = 1., 0.3
# E2, nu2 = E1/100., nu1
# E_nu_tuples = [(E1, nu1), (E2, nu2)]
# phy_subdomains = panto_test.phy_surf
# material_dict = dict()
# for coeff, subdo in zip(E_nu_tuples, phy_subdomains):
#     material_dict[subdo.tag] = mat.Material(coeff[0], coeff[1], 'cp')
# rve = prt.Fenics2DRVE.gmsh_2_Fenics_2DRVE(panto_test, material_dict)
# # #* Test ok


# L, t = 1, 0.05
# a = L-3*t
# aux_test = mesh_generate_2D.Gmsh2DRVE.auxetic_square(L, a, t, nb_cells=(2, 3), soft_mat=True, name='aux_test_mou')
# aux_test.main_mesh_refinement((0.1,0.5),(0.03,0.3),False)
# aux_test.soft_mesh_refinement((0.1,0.5),(0.03,0.3),False)
# aux_test.mesh_generate()
# run(f"gmsh {aux_test.name}.msh &",shell=True, check=True)
# E1, nu1 = 1., 0.3
# E2, nu2 = E1/100., nu1
# E_nu_tuples = [(E1, nu1), (E2, nu2)]
# phy_subdomains = aux_test.phy_surf
# material_dict = dict()
# for coeff, subdo in zip(E_nu_tuples, phy_subdomains):
#     material_dict[subdo.tag] = mat.Material(coeff[0], coeff[1], 'cp')
# rve = prt.Fenics2DRVE.gmsh_2_Fenics_2DRVE(aux_test, material_dict)

def test_global_area_2D():
    """FenicsPart method global_aera, for a 2D part"""
    L_x, L_y = 10., 4.
    mesh = fe.RectangleMesh(fe.Point(0.0, 0.0), fe.Point(L_x, L_y), 10, 10)
    material = {'0':mat.Material(1., 0.3, 'cp')}
    dimensions = np.array(((L_x, 0.),(0., L_y)))
    rect_part = part.FenicsPart(mesh, materials=material, subdomains=None,
        global_dimensions=dimensions, facet_regions=None)
    assert rect_part.global_area() == approx(40.,rel=1e-10)

def test_mat_area():
    """FenicsPart method global_aera, for 2D parts created from a gmsh mesh and from a FEniCS mesh"""
    L_x, L_y = 4., 5.
    H = 1.
    size = 0.5
    rectangle = mshr.Rectangle(fe.Point(0., 0), fe.Point(L_x, L_y))
    hole =  mshr.Rectangle(fe.Point(L_x/2-H/2, L_y/2-H/2),
        fe.Point(L_x/2+H/2, L_y/2+H/2))
    domain = rectangle - hole
    domain.set_subdomain(1, rectangle)
    mesh = mshr.generate_mesh(domain, size)
    dimensions = np.array(((L_x, 0.),(0., L_y)))
    material = {'0':mat.Material(1., 0.3, 'cp')}
    rect_part = part.FenicsPart(mesh, materials=material, subdomains=None,
        global_dimensions=dimensions, facet_regions=None)
    assert rect_part.mat_area() == (L_x*L_y - H**2)

    name = "test_mat_area"
    local_dir = Path(__file__).parent
    mesh_file = local_dir.joinpath(name+'.msh')
    gmsh.model.add(name)
    vertices = [(0., 0.), (0., L_y), (L_x, L_y), (L_x, 0.)]
    contour = geo.LineLoop([geo.Point(np.array(c)) for c in vertices], False)
    surface = geo.PlaneSurface(contour)
    cut_vertices = list()
    for local_coord in [(H, 0., 0.), (0., H, 0.), (-H, 0., 0.), (0., -H, 0.)]:
        vertex = geo.translation(contour.vertices[2], 
            np.array(local_coord))
        cut_vertices.append(vertex)
    cut_surface = geo.PlaneSurface(geo.LineLoop(cut_vertices,False))
    for s in [surface, cut_surface]:
        s.add_gmsh()
    factory.synchronize()
    surface, = geo.AbstractSurface.bool_cut(surface, cut_surface)
    factory.synchronize()
    for dim_tag in model.getEntities(2):
        if not dim_tag[1] == surface.tag:
            model.removeEntities(dim_tag, True)
    charact_field = ho_homog.mesh_tools.MathEvalField("0.1")
    ho_homog.mesh_tools.set_background_mesh(charact_field)
    geo.set_gmsh_option('Mesh.SaveAll', 1)
    model.mesh.generate(2)
    gmsh.write(str(mesh_file))
    cmd = f"dolfin-convert {mesh_file} {mesh_file.with_suffix('.xml')}"
    run(cmd, shell=True, check=True)
    mesh = fe.Mesh(str(mesh_file.with_suffix('.xml')))
    dimensions = np.array(((L_x, 0.),(0., L_y)))
    material = {'0':mat.Material(1., 0.3, 'cp')}
    rect_part = part.FenicsPart(mesh, materials=material, subdomains=None,
        global_dimensions=dimensions, facet_regions=None)
    assert rect_part.mat_area() == approx(L_x*L_y - H*H/2)
    geo.reset()

def test_mat_without_dictionnary():
    """FenicsPart instance initialized with only one instance of Material"""
    L_x, L_y = 1, 1
    mesh = fe.RectangleMesh(fe.Point(0.0, 0.0), fe.Point(L_x, L_y), 10, 10)
    dimensions = np.array(((L_x, 0.),(0., L_y)))
    E, nu = 1, 0.3
    material = mat.Material(E, nu, 'cp')
    rect_part = part.FenicsPart(mesh, materials=material, subdomains=None,
        global_dimensions=dimensions, facet_regions=None)
    elem_type = 'CG'
    degree = 2
    strain_fspace = fe.FunctionSpace(mesh,
        fe.VectorElement(elem_type, mesh.ufl_cell(), degree, dim=3),
        )
    strain = fe.project(fe.Expression(
        ("1.0+x[0]*x[0]", "0", "1.0"),
        degree=2), strain_fspace)
    stress = rect_part.sigma(strain)
    energy = fe.assemble(fe.inner(stress, strain) * fe.dx(rect_part.mesh))
    energy_theo = E/(1 + nu) * (1 + 28/(15*(1-nu)))
    assert energy == approx(energy_theo, rel=1e-13)

def test_2_materials():
    """FenicsPart instance initialized with only one instance of Material in the materials dictionnary"""
    L_x, L_y = 1, 1
    size = 0.1
    mesh = fe.RectangleMesh(fe.Point(-L_x, -L_y), fe.Point(L_x, L_y), 20, 20)
    dimensions = np.array(((2*L_x, 0.),(0., 2*L_y)))
    subdomains = fe.MeshFunction("size_t", mesh, 2)
    class Right_part(fe.SubDomain):
        def inside(self, x, on_boundary):
            return x[0] >= 0 - fe.DOLFIN_EPS
    subdomain_right = Right_part()
    subdomains.set_all(0)
    subdomain_right.mark(subdomains, 1)
    E_1, E_2, nu = 1, 3, 0.3
    materials = {
        0: mat.Material(1, 0.3, 'cp'),
        1: mat.Material(3, 0.3, 'cp')
    }
    rect_part = part.FenicsPart(mesh, materials, subdomains, dimensions)
    elem_type = 'CG'
    degree = 2
    strain_fspace = fe.FunctionSpace(
        mesh,
        fe.VectorElement(elem_type, mesh.ufl_cell(), degree, dim=3),
        )
    strain = fe.project(fe.Expression(
        ("1.0+x[0]*x[0]", "0", "1.0"),
        degree=2), strain_fspace)
    stress = rect_part.sigma(strain)
    energy = fe.assemble(fe.inner(stress, strain) * fe.dx(rect_part.mesh))
    energy_theo = 2*((E_1+E_2)/(1 + nu) * (1 + 28/(15*(1-nu))))
    assert energy == approx(energy_theo, rel=1e-13)

def test_get_domains_gmsh(plots=False):
    """ Get subdomains and partition of the boundary from a .msh file. """
    name = "test_domains"
    local_dir = Path(__file__).parent
    mesh_file = local_dir.joinpath(name+'.msh')
    gmsh.model.add(name)
    L_x, L_y = 2., 2.
    H = 1.
    vertices = [(0., 0.), (0., L_y), (L_x, L_y), (L_x, 0.)]
    contour = geo.LineLoop([geo.Point(np.array(c)) for c in vertices], False)
    surface = geo.PlaneSurface(contour)
    inclusion_vertices = list()
    for coord in [(H/2, -H/2, 0.), (H/2, H/2, 0.), (-H/2, H/2, 0.), (-H/2, -H/2, 0.)]:
        vertex = geo.translation(geo.Point((L_x/2, L_y/2)), coord)
        inclusion_vertices.append(vertex)
    inclusion = geo.PlaneSurface(geo.LineLoop(inclusion_vertices,False))
    for s in [surface, inclusion]:
        s.add_gmsh()
    factory.synchronize()
    stiff_s, = geo.AbstractSurface.bool_cut(surface, inclusion)
    factory.synchronize()
    soft_s, = geo.AbstractSurface.bool_cut(surface, stiff_s)
    factory.synchronize()
    domains = {
        'stiff': geo.PhysicalGroup(stiff_s, 2),
        'soft': geo.PhysicalGroup(soft_s, 2)
    }
    boundaries = {
        'S': geo.PhysicalGroup(surface.ext_contour.sides[0], 1),
        'W': geo.PhysicalGroup(surface.ext_contour.sides[1], 1),
        'N': geo.PhysicalGroup(surface.ext_contour.sides[2], 1),
        'E': geo.PhysicalGroup(surface.ext_contour.sides[3], 1),
    }
    for group in domains.values():
        group.add_gmsh()
    for group in boundaries.values():
        group.add_gmsh()
    charact_field = ho_homog.mesh_tools.MathEvalField("0.1")
    ho_homog.mesh_tools.set_background_mesh(charact_field)
    geo.set_gmsh_option('Mesh.SaveAll', 0)
    model.mesh.generate(1)
    model.mesh.generate(2)
    gmsh.model.mesh.removeDuplicateNodes()
    gmsh.write(str(mesh_file))
    E_1, E_2, nu = 1, 3, 0.3
    materials = {
        domains['soft'].tag: mat.Material(1, 0.3, 'cp'),
        domains['stiff'].tag: mat.Material(1, 0.3, 'cp')
    }
    test_part = part.FenicsPart.file_2_FenicsPart(
        str(mesh_file), materials, subdomains_import=True)
    assert test_part.mat_area() == approx(L_x*L_y)
    elem_type = 'CG'
    degree = 2
    V = fe.VectorFunctionSpace(test_part.mesh, elem_type, degree)
    W = fe.FunctionSpace(
        test_part.mesh,
        fe.VectorElement(elem_type, test_part.mesh.ufl_cell(), degree, dim=3),
    )   
    boundary_conditions = {
        boundaries['N'].tag: fe.Expression(("x[0]-1", "1"), degree=1),
        boundaries['S'].tag: fe.Expression(("x[0]-1", "-1"), degree=1),
        boundaries['E'].tag: fe.Expression(("1", "x[1]-1"), degree=1),
        boundaries['W'].tag: fe.Expression(("-1", "x[1]-1"), degree=1)
        }
    bcs = list()
    for tag, val in boundary_conditions.items():
        bcs.append(fe.DirichletBC(V, val, test_part.facet_regions, tag))
    ds = fe.Measure(
        'ds', domain=test_part.mesh,
        subdomain_data=test_part.facet_regions
    )
    v = fe.TestFunctions(V)
    u = fe.TrialFunctions(V)
    F = fe.inner(
            test_part.sigma(test_part.epsilon(u)),
            test_part.epsilon(v)
            ) * fe.dx
    a, L = fe.lhs(F), fe.rhs(F)
    u_sol = fe.Function(V)
    fe.solve(a == L, u_sol, bcs)
    strain = fe.project(test_part.epsilon(u_sol),W)
    if plots:
        import matplotlib.pyplot as plt
        plt.figure()
        plot = fe.plot(u_sol)
        plt.colorbar(plot)
        plt.figure()
        plot = fe.plot(strain[0])
        plt.colorbar(plot)
        plt.figure()
        plot = fe.plot(strain[1])
        plt.colorbar(plot)
        plt.figure()
        plot = fe.plot(strain[2])
        plt.colorbar(plot)
        plt.show()
    error = fe.errornorm(strain, fe.Expression(("1","1","0"),degree=0),degree_rise=3, mesh=test_part.mesh)
    assert error == approx(0,abs=5e-13)
    geo.reset()