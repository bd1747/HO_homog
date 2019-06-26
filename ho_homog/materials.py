"""
Created on Wed Apr 18 13:39:42 2018

@author: baptiste
"""


import logging
import math
import warnings

import dolfin as fe
import numpy as np

logger = logging.getLogger(__name__)

PLANE_IDX = np.array([0, 1, 5])


class Material(object):
    def __init__(self, E, nu, cas_elast="cp"):
        """ Définir un nouveau comportement élastique, exclusivement isotrope pour le moment.
         cas_elast : Choisir entre 'cp' contraintes planes, 'dp' déformations planes ou '3D'.
        """

        self.E = E
        self.nu = nu
        self.G = self.E / (2.0 * (1 + self.nu))
        self.cas_elast = cas_elast

        self.S_3D = np.array(
            (
                (1.0 / self.E, -self.nu / self.E, -self.nu / self.E, 0.0, 0.0, 0.0),
                (-self.nu / self.E, 1.0 / self.E, -self.nu / self.E, 0.0, 0.0, 0.0),
                (-self.nu / self.E, -self.nu / self.E, 1.0 / self.E, 0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0, 1.0 / (2.0 * self.G), 0.0, 0.0),
                (0.0, 0.0, 0.0, 0.0, 1.0 / (2.0 * self.G), 0.0),
                (0.0, 0.0, 0.0, 0.0, 0.0, 1.0 / (2.0 * self.G)),
            )
        )
        self.C_3D = np.linalg.inv(self.S_3D)
        self.set_elast[cas_elast](self)

    def set_cp(self):
        self.S = self.S_3D[PLANE_IDX[:, None], PLANE_IDX]
        self.C = np.linalg.inv(self.S)
        self.cas_elast = "cp"

    def set_dp(self):
        self.C = self.C_3D[PLANE_IDX[:, None], PLANE_IDX]
        self.S = np.linalg.inv(self.C)
        self.cas_elast = "dp"

    def set_3D(self):
        self.S = self.S_3D
        self.C = self.C_3D
        self.cas_elast = "3D"

    # Dictionnaire des méthodes set
    set_elast = {"cp": set_cp, "dp": set_dp, "3D": set_3D}

    def get_S(self):
        """ Renvoie la matrice de souplesse 3D, cp ou dp dans un format adapté pour FEniCs."""
        return fe.Constant(self.S)

    def get_C(self):
        """ Renvoie la matrice de raideur 3D, cp ou dp dans un format adapté pour FEniCs."""
        # return fe.Constant(self.C)
        return self.C  # Correction, plus rapide

    def get_E(self):
        return self.E

    def get_nu(self):
        return self.nu


class StiffnessComponent(fe.UserExpression):
    """ FEniCS Expression that represent one component of the stiffness tensor on a mesh that contain several subdomains."""

    def __init__(self, cell_function, mat_dict, i, j, **kwargs):
        self.cell_function = cell_function
        self.mat_dict = mat_dict
        self.i = i
        self.j = j
        super().__init__(degree=kwargs["degree"])
        # ? Info : https://docs.python.org/fr/3/library/functions.html#super,
        # ? and http://folk.uio.no/kent-and/hpl-fem-book/doc/pub/book/pdf/fem-book-4print-2up.pdf

    def eval_cell(self, values, x, cell):
        subdomain_id = self.cell_function[cell.index]
        values[0] = self.mat_dict[subdomain_id].get_C()[self.i, self.j]

    def value_shape(self):
        return ()


class ElasticityParamOnSubdomain(fe.UserExpression):
    """ FEniCS Expression that represent one component of the stiffness tensor on a mesh that contain several subdomains."""

    def __init__(self, cell_function, mat_dict, elasticity_param, **kwargs):
        """[summary]

        Parameters
        ----------
        elasticity_param : str
            Choix du paramètre que l'on souhaite récupérer.
        """

        self.cell_function = cell_function
        self.mat_dict = mat_dict
        self.elasticity_param = elasticity_param
        super().__init__(degree=kwargs["degree"])

    def eval_cell(self, values, x, cell):
        subdomain_id = self.cell_function[cell.index]
        values[0] = getattr(self.mat_dict[subdomain_id], self.elasticity_param)

    def value_shape(self):
        return ()


def mat_per_subdomains(cell_function, mat_dict, topo_dim):
    """Définir un comportement hétérogène par sous domaine.

    Parameters
    ----------
    cell_function : FEniCS Mesh function
        Denified on the cells. Indicates the subdomain to which belongs each cell.
    mat_dict : dictionnary
        [description]
    topo_dim : int
        [description]

    Returns
    -------
    C_per
        FEniCS matrix that represents the stiffness inside the RVE.
    """

    C = []
    nb_val = int(topo_dim * (topo_dim + 1) / 2)
    for i in range(nb_val):
        Cj = []
        for j in range(nb_val):
            Cj = Cj + [StiffnessComponent(cell_function, mat_dict, i, j, degree=0)]
        C = C + [Cj]
    C_per = fe.as_matrix(C)
    return C_per


def epsilon(u):
    return fe.as_vector(
        (u[0].dx(0), u[1].dx(1), (u[0].dx(1) + u[1].dx(0)) / math.sqrt(2))
    )


def sigma(C, eps):
    return C * eps


def strain_cross_energy(sig, eps, mesh, area):
    """
    Calcul de l'energie croisée des champs de contrainte sig et de deformation eps.
    """

    return fe.assemble(fe.inner(sig, eps) * fe.dx(mesh)) / area


def cross_energy(sig, eps, mesh):
    """
    Calcul de l'energie croisée des champs de contrainte sig et de deformation eps.
    """
    return fe.assemble(fe.inner(sig, eps) * fe.dx(mesh))


# def energy_norm()
#     return fe.assemble(fe.inner(sig, eps) * fe.dx(mesh))/area

# code energy_norm : https://bitbucket.org/fenics-project/ufl/src/master/ufl/formoperators.py


class StiffnessComponentLevelSet(fe.UserExpression):
    def __init__(self, level_set, mat_dict, i, j, threshold: float = 0.0, **kwargs):
        self.level_set = level_set
        self.mat_dict = mat_dict
        self.i = i
        self.j = j
        self.threshold = threshold
        super().__init__(degree=kwargs["degree"])

    def eval_cell(self, values, x, cell):
        stiffness = 1 if self.level_set.eval(x) >= 0 else 0
        values[0] = self.mat_dict[stiffness].get_C()[self.i, self.j]

    def value_shape(self):
        return ()


def C_from_levelset(level_set, mat_dict, topo_dim, threshold: float = 0.0):
    """Définir un comportement hétérogène par sous domaine.

    Parameters
    ----------
    level_set : meshfunction
        Denified on the cells. Scalar values
    mat_dict : dictionnary
        2 instances of Material.
        keys, values :
            - "0" : infinitely "soft" material;
            - "1" : material with "standard" elastic properties
    topo_dim : int
        Dimension of the mesh (2D, 3D)
    threshold : float
        Threshold value for the interpretation of the levelset

    Returns
    -------
    stiffness_matrix
        FEniCS matrix that represents the stiffness inside the RVE.
    """

    C = []
    nb_val = int(topo_dim * (topo_dim + 1) / 2)
    for i in range(nb_val):
        Cj = []
        for j in range(nb_val):
            Cj = Cj + [StiffnessComponent(cell_function, mat_dict, i, j, degree=0)]
        C = C + [Cj]
    stiffness_matrix = fe.as_matrix(C)
    return stiffness_matrix
