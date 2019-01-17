"""
Created on Wed Apr 18 13:39:42 2018

@author: baptiste
"""


import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import warnings
import dolfin as fe

import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("materials") #http://sametmax.com/ecrire-des-logs-en-python/
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s')
file_handler = RotatingFileHandler('activity.log', 'a', 1000000, 1)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler) #Pour écriture d'un fichier log
formatter = logging.Formatter('%(levelname)s :: %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

PLANE_IDX = np.array([0, 1, 5])


class Material(object):

    def __init__(self, E, nu, cas_elast='cp'):
        """ Définir un nouveau comportement élastique, exclusivement isotrope pour le moment.
         cas_elast : Choisir entre 'cp' contraintes planes, 'dp' déformations planes ou '3D'.
        """

        self.E = E
        self.nu = nu
        self.G = self.E / (2.*(1 + self.nu))
        self.cas_elast = cas_elast

        self.S_3D = np.array((
            (1./self.E, -self.nu/self.E, -self.nu/self.E, 0., 0., 0.),
            (-self.nu/self.E, 1./self.E, -self.nu/self.E, 0., 0., 0.),
            (-self.nu/self.E, -self.nu/self.E, 1./self.E, 0., 0., 0.),
            (0., 0., 0., 1./(2.*self.G), 0., 0.),
            (0., 0., 0., 0., 1./(2.*self.G), 0.),
            (0., 0., 0., 0., 0., 1./(2.*self.G))
            ))
        self.C_3D = np.linalg.inv(self.S_3D)
        self.set_elast[cas_elast](self)

    def set_cp(self):
        self.S = self.S_3D[PLANE_IDX[:, None], PLANE_IDX]
        self.C = np.linalg.inv(self.S)
        self.cas_elast = 'cp'

    def set_dp(self):
        self.C = self.C_3D[PLANE_IDX[:, None], PLANE_IDX]
        self.S = np.linalg.inv(self.C)
        self.cas_elast = 'dp'
    
    def set_3D(self):
        self.S = self.S_3D
        self.C = self.C_3D
        self.cas_elast = '3D'

    # Dictionnaire des méthodes set
    set_elast = {"cp": set_cp, "dp": set_dp, "3D": set_3D}
    
    def get_S(self):
        """ Renvoie la matrice de souplesse 3D, cp ou dp dans un format adapté pour FEniCs."""
        return fe.Constant(self.S)
    
    def get_C(self):
        """ Renvoie la matrice de raideur 3D, cp ou dp dans un format adapté pour FEniCs."""
        # return fe.Constant(self.C)
        return self.C  #  Correction, plus rapide


class StiffnessComponent(fe.UserExpression):
    """ FEniCS Expression that represent one component of the stiffness tensor on a mesh that contain several subdomains."""
    def __init__(self, cell_function, mat_dict, i, j, **kwargs):
        self.cell_function = cell_function
        self.mat_dict = mat_dict
        self.i = i
        self.j = j
        super().__init__(degree=kwargs["degree"])
        #? Info : https://docs.python.org/fr/3/library/functions.html#super, 
        #? and http://folk.uio.no/kent-and/hpl-fem-book/doc/pub/book/pdf/fem-book-4print-2up.pdf

    def eval_cell(self, values, x, cell):
        subdomain_id = self.cell_function[cell.index]
        values[0] = self.mat_dict[subdomain_id].get_C()[self.i, self.j]

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
    nb_val = int(topo_dim * (topo_dim + 1)/2)
    for i in range(nb_val):
        Cj = []
        for j in range(nb_val):
            Cj = Cj + [StiffnessComponent(cell_function, mat_dict, i, j, degree=0)]
        C = C + [Cj]
    C_per = fe.as_matrix(C)
    return C_per


class MaterialsPerDomains(object):
    """
    Définir un comportement hétérogène par sous domaine.
    """
    class StiffnessComponent(fe.UserExpression):
        def __init__(self, cell_function, mat_dict, i, j, **kwargs):
            warnings.warn("Deprecated. Should use the mat_per_subdomains function instead.", DeprecationWarning)
            #! LA on rencontre une erreure majeure.
            #! Message :
            #! [Previous line repeated 324 more times] RecursionError: maximum recursion depth exceeded
            #? Solution ? Regarder la solution de substitution à fe.Expression mise en place dans FEniCS 2018
            self.cell_function = cell_function
            self.mat_dict = mat_dict
            self.i = i
            self.j = j
            super().__init__(degree=kwargs["degree"])
            #? Info : https://docs.python.org/fr/3/library/functions.html#super, 
            #? and http://folk.uio.no/kent-and/hpl-fem-book/doc/pub/book/pdf/fem-book-4print-2up.pdf

        def eval_cell(self, values, x, cell):
            subdomain_id = self.cell_function[cell.index]
            values[0] = self.mat_dict[subdomain_id].get_C()[self.i, self.j]

    def __init__(self, cell_function, mat_dict, topo_dim, **kwargs): #Todo : supprimer kwargs ?
        self.cell_function = cell_function
        self.mat_dict = mat_dict

        C = []
        print(f"topo dim : {topo_dim} nb de valeurs pour i et j : {topo_dim * (topo_dim + 1)/2}")
        nb_val = int(topo_dim * (topo_dim + 1)/2)
        for i in range(nb_val):
            Cj = []
            for j in range(nb_val):
                Cj = Cj + [self.StiffnessComponent(self.cell_function, mat_dict, i, j, degree=0)]
            C = C + [Cj]
        logger.debug("C_per before conversion %s", C)
        self.C_per = fe.as_matrix(C) #TODO : remplacer classe par une fonction

### TO DO : essayer de sortir cette nested définition de classe de la classe MatDomains2
          
    #     # Assembling the constitutive matrix
    #     Ci = []
    #     for i in range(self.dim * (self.dim + 1)/2):
    #         Cj = []
    #         for j in range(self.dim * (self.dim + 1)/2):
    #             Cj = Cj + [StiffnessComponent(self.materials, listOfMaterials,i,j, degree=0)]
    #         Ci = Ci + [Cj]
    # self.Cper = fe.as_matrix(Ci)


def epsilon(u):
    return fe.as_vector((u[0].dx(0), u[1].dx(1), (u[0].dx(1) + u[1].dx(0))/math.sqrt(2)))

def sigma(C, eps):
    return C * eps
    
def strain_cross_energy(sig, eps, mesh, area):
    """
    Calcul de l'energie croisée des champs de contrainte sig et de deformation eps.
    """
    
    return fe.assemble(fe.inner(sig, eps) * fe.dx(mesh))/area