# coding: utf-8
"""
Created on 17/11/2021
@author: baptiste

Schéma d'homogénéisation vers milieu de second gradient, appliqué à RVE homogène.
Avec D, raideur de 2d gradient = F - (G + G.T)


"""

import logging
from pathlib import Path

import dolfin as fe
import gmsh
import numpy as np
from pytest import approx

from ho_homog import geometry, homog2d, materials, part
from ho_homog.mesh_generate import Gmsh2DRVE
from ho_homog.toolbox_gmsh import msh_conversion

TEMP_DIR = Path(__file__).parent / "_temp_"
if not TEMP_DIR.exists():
    TEMP_DIR.mkdir()

logger = logging.getLogger("Test_homog2d")
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    format="%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.DEBUG,
    force=True,
)
logging.getLogger().setLevel(logging.DEBUG)

np.set_printoptions(suppress=False, floatmode="fixed", precision=6, linewidth=200)


def FG_to_D(consti_tens_dict):
    """
    Opération d'intégration par partie formelle du schéma d'homogénéisation pour la troncature à l'ordre 2.

    ----------
    consti_tens_dict : dictionnary
        [description]

    # >>> consti_tens = {\
    #     "E": {\
    #        "EGGbis": np.array([[-2.08e-06, 1.37e-05, 3.26e-07, 1.55e-06, 7.07e-07, 3.68e-06, 4.51e-06, 1.04e-05, 4.88e-05, -6.13e-06, -9.50e-06, -2.02e-06],\
    #                [0.00, 0.02, 2.74e-06, 1.20e-05, 1.05e-05, -0.01, -1.56e-06, 3.33e-06, 0.01, 0.00, 0.00, -1.33e-06],\
    #                [7.10e-08,-1.90e-07, 2.68e-08, 4.88e-06, 3.01e-06, 4.54e-07, 2.09e-05, 5.30e-05, 7.17e-07, 8.18e-08, 7.33e-08, 8.84e-06]])\
    #        },\
    #    "EG": {\
    #        "EG": np.array([[0.02, 0.02, -3.40e-05, 1.22e-05, 1.43e-05, -0.00],\
    #                [0.02, 0.04,-3.65e-05, 1.79e-05, 1.95e-05,-0.00],\
    #                [-3.40e-05, -3.65e-05, 0.09, 0.01, 0.00, -7.79e-08],\
    #                [1.22e-05, 1.79e-05, 0.01, 0.12, 0.10,-1.34e-06],\
    #                [ 1.43e-05, 1.95e-05, 0.00, 0.10, 0.10,-1.28e-06],\
    #                [-0.00, -0.00, -7.79e-08, -1.34e-06, -1.28e-06, 2.52e-05]]),\
    #        }\
    #    }
    # >>> FG_to_D(consti_tens)
    # array([[ 2.000416e-02,  1.998630e-02, -3.439700e-05,  6.140000e-06,  1.515300e-05, -2.458000e-05],
    #        [ 1.998630e-02,  0.000000e+00, -3.905000e-05, -4.500000e-06,  5.670000e-06,  9.947000e-03],
    #        [-3.439700e-05, -3.905000e-05,  8.999995e-02,  9.946320e-03, -1.000301e-02, -1.248900e-06],
    #        [ 6.140000e-06, -4.500000e-06,  9.946320e-03,  1.200123e-01,  1.000095e-01,  5.982000e-07],
    #        [ 1.515300e-05,  5.670000e-06, -1.000301e-02,  1.000095e-01,  1.000000e-01, -2.330000e-08],
    #        [-2.458000e-05,  9.947000e-03, -1.248900e-06,  5.982000e-07, -2.330000e-08,  7.520000e-06]])
    """
    # TODO: A mettre autre part, avec homog2d
    # TODO : Faire test + simple

    F = consti_tens_dict["EG"]["EG"].copy()
    try:
        G = consti_tens_dict["E"]["EGG"].copy()
    except KeyError:
        msg = "Constitutive tensor E <-> EGG not found. The one for E <-> EGGbis will be used."
        logger.warning(msg)
        G = consti_tens_dict["E"]["EGGbis"].copy()
    G_6x6 = np.vstack((G[:, :6], G[:, 6:]))
    logger.debug("Type de G : %s, shape : %s", type(G), G.shape)
    logger.debug("Nouvelle forme de G (shape %s) : \n %s", G_6x6.shape, G_6x6)
    D = F - G_6x6 - G_6x6.T
    logger.debug("F is : \n %s", F)
    logger.debug("G is : \n %s", G)
    logger.debug("D is : \n %s", D)
    return D


def _mesh_homogeneous_cell(cell_vect, mesh_path):
    """Generate a  simple mesh for a homogeneous cell.
    cell_vect: np.array 2x2  colonnes = vecteurs periodicité
    """
    name = mesh_path.stem
    geometry.init_geo_tools()
    geometry.set_gmsh_option("Mesh.MshFileVersion", 4.1)
    # Mesh.Algorithm = 6; Frontal - Delaunay for 2D meshes
    geometry.set_gmsh_option("Mesh.Algorithm", 6)
    geometry.set_gmsh_option("Mesh.MeshSizeMin", 0.05)
    geometry.set_gmsh_option("Mesh.MeshSizeMax", 0.05)

    rve = Gmsh2DRVE([], cell_vect, (1, 1), np.zeros(2), [], False, name)
    rve.mesh_generate()
    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()
    gmsh.write(str(mesh_path))
    mesh_path = msh_conversion(mesh_path, ".xdmf")
    geometry.reset()
    return mesh_path


def _save_all_localization_fields(xdmf_path: Path, localization_dicts: list):
    """Sauvegarde de tous les champs de localisation dans un .xdmf
    localization_dicts : list de dictionnaires."""
    with fe.XDMFFile(str(xdmf_path)) as fo:
        fo.parameters["functions_share_mesh"] = True
        for d in localization_dicts:
            for k, fields in d.items():
                for f in fields:
                    logger.debug(f"Saving field {f.name()}")
                    fo.write(f, 0.0)
    return None


E, nu = 1.0, 0.3
ref_homogeneous_material = materials.Material(E, nu, "cp")


def test_homogeneous_rectangular_cell():
    """Test élémentaire  : Homogénéisation d'une cellule homogène rectangulaire."""
    logger.debug("Start test_homogeneous_rectangular_cell")
    # Generation du maillage :
    cell_vect = np.array([[1.0, 0.0], [0.0, 2.0]])  # colonnes = vecteurs periodicité
    name = "rectangular_homogeneous_cell"
    mesh_path = _mesh_homogeneous_cell(cell_vect, TEMP_DIR / f"{name}.msh")

    # Definition du RVE :
    rve = part.Fenics2DRVE.rve_from_file(mesh_path, cell_vect, ref_homogeneous_material)
    hom_model = homog2d.Fenics2DHomogenization(rve, element=("CG", 2))
    output = hom_model.homogenizationScheme("EGG")
    loc_u, loc_s, loc_e, consti_tensors = output

    # Verification des tenseurs de raideur :
    stiff_E_ref = ref_homogeneous_material.get_C()
    stiff_E_result = consti_tensors["E"]["E"]
    stiff_K_before_int = consti_tensors["EG"]["EG"]
    stiff_K_after_int = FG_to_D(consti_tensors)
    coupling_E_K = consti_tensors["E"]["EG"]

    logger.debug(f"Results C: \n {stiff_E_result} ")
    logger.debug(f"Results K<->K, D:\n {stiff_K_after_int} ")
    logger.debug(f"Results K<->K, F:\n {stiff_K_before_int} ")
    logger.debug(f"Results coupling K<->E: \n {coupling_E_K}")
    stiff_K_ref = np.zeros((6, 6))
    coupling_E_K_ref = np.zeros((3, 6))
    # Choix d'une tolérance absolue, liée aux valeurs de la raideur matériau:
    # Comme Module de Young E = 1, atol=1e-6 semble raisonnable.
    assert stiff_E_result == approx(stiff_E_ref, rel=1e-6, abs=1e-6)
    # Pour stiff K<->K et E<->K, relative tolerance is irrelevant as 0 is expected.
    # Absolute tolerance assez stricte, car ça devrait être exactement 0 : atol=1e-15
    assert stiff_K_before_int == approx(stiff_K_ref, abs=1e-15)
    assert stiff_K_after_int == approx(stiff_K_ref, abs=1e-15)
    assert coupling_E_K == approx(coupling_E_K_ref, abs=1e-15)

    # Enregistrement des champs, pour debug :
    field_path = mesh_path.with_name(f"{mesh_path.stem}_fields.xdmf")
    _save_all_localization_fields(field_path, [loc_u, loc_s, loc_e])
    logger.debug("End test_homogeneous_rectangular_cell")


def test_homogeneous_tilted_cell():
    """Test élémentaire  : Homogénéisation d'une cellule en forme de losange."""
    logger.debug("Start test_homogeneous_tilted_cell")
    # Generation du maillage :
    cell_vect = np.array([[1.0, 1 / 2], [0, np.sqrt(3) / 2]])
    name = "tilted_homogeneous_cell"
    mesh_path = _mesh_homogeneous_cell(cell_vect, TEMP_DIR / f"{name}.msh")

    rve = part.Fenics2DRVE.rve_from_file(mesh_path, cell_vect, ref_homogeneous_material)
    hom_model = homog2d.Fenics2DHomogenization(rve, element=("CG", 2))
    loc_u, loc_s, loc_e, consti_tensors = hom_model.homogenizationScheme("EGG")

    # Verification des tenseurs de raideur :
    stiff_E_ref = ref_homogeneous_material.get_C()
    c = consti_tensors["E"]["E"]
    f, d = consti_tensors["EG"]["EG"], FG_to_D(consti_tensors)
    b = consti_tensors["E"]["EG"]
    logger.debug(f"Results C: \n {c} ")
    logger.debug(f"Results K<->K, D:\n {d} ")
    logger.debug(f"Results K<->K, F:\n {f} ")
    logger.debug(f"Results coupling K<->E: \n {b}")
    assert c == approx(stiff_E_ref, rel=1e-6, abs=1e-6)
    assert f == approx(np.zeros((6, 6)), abs=1e-15)
    assert d == approx(np.zeros((6, 6)), abs=1e-15)
    assert b == approx(np.zeros((3, 6)), abs=1e-15)

    # Enregistrement des champs, pour debug :
    field_path = mesh_path.with_name(f"{mesh_path.stem}_fields.xdmf")
    _save_all_localization_fields(field_path, [loc_u, loc_s, loc_e])
    logger.debug("End test_homogeneous_tilted_cell")


# if __name__ == "__main__":
# test_homogeneous_rectangular_cell()
# test_homogeneous_tilted_cell()
