"""
Created on Wed Apr 18 13:39:42 2018

@author: baptiste
"""


import copy
import math
import matplotlib.pyplot as plt
import numpy as np

import dolfin as fe


class Material(object):
    plane_idx = np.array([0, 1, 5])

    def __init__(self, E, nu, cas_elast='cp'):
        """ Définir un nouveau comportement élastique, exclusivement isotrope pour le moment. 
         cas_elast : Choisir entre 'cp' contraintes planes, 'dp' déformations planes ou '3D'.
        """
#        TO DO : stocker les matrices 3D, une clé, et les fe.matrix pour y accéder rapidement

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
        self.S = self.S_3D[Material.plane_idx[:, None], Material.plane_idx]
        self.C = np.linalg.inv(self.S)
        self.cas_elast = 'cp'

    def set_dp(self):
        self.C = self.C_3D[Material.plane_idx[:, None], Material.plane_idx]
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
        return self.C  #  Correction pour revenir au code d'Arthur plus rapide (avec MatDomain2)


class MaterialsPerDomains(object):
    """
    Définir un comportement hétérogène par sous domaine.
    """

    def __init__(self, cell_function, mat_dict, topo_dim, **kwargs):
        self.cell_function = cell_function
        self.mat_dict = mat_dict

        C = []
        print(f"topo dim : {topo_dim} nb de valeurs pour i et j : {topo_dim * (topo_dim + 1)/2}")
        for i in range(topo_dim * (topo_dim + 1)/2):
            Cj = []
            for j in range(topo_dim * (topo_dim + 1)/2):
                Cj = Cj + [self.StiffnessComponent(self.cell_function, mat_dict, i, j, degree=0)]
            C = C + [Cj]
        self.C_per = fe.as_matrix(C)

### TO DO : essayer de sortir cette nested définition de classe de la classe MatDomains2
    class StiffnessComponent(fe.Expression):
                def __init__(self, cell_function, mat_dict, i, j, **kwargs):
                    self.cell_function = cell_function
                    self.mat_dict = mat_dict
                    self.i = i
                    self.j = j
            
                def eval_cell(self, values, x, cell):
                    subdomain_id = self.cell_function[cell.index]
                    values[0] = self.mat_dict[subdomain_id].get_C()[self.i, self.j]
          
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