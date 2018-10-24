# -*- coding: utf-8 -*-
"""
Created on 22/10/2018
@author: baptiste

Définition de classes d'objets géométriques et de fonctions permettant de créer un modèle géométrique de RVE dans gmsh.

Outils pour controler la partie maillage dans gmsh, en complément des outils utiles pour la construction du modèle géoémtrique. 
Deux fonctionnalités : 
        - Raffinement du maillage autour de points d'intérêts, en utilisant des champs scalaire pour prescire la taille caractéristiques des éléments. 
        - Maillage "périodique" : identique sur les parties du bords du RVE en vis à vis.
"""

import copy
import os

import matplotlib.cm as pltcm
import matplotlib.pyplot as plt
import numpy as np

import geometry as geo
import gmsh

# nice shortcuts
model = gmsh.model
factory = model.occ
api_field = model.mesh.field

class Field(object):
    """
    Class mère (=générique) pour représenter les champs scalaires utilisés pour prescrire la taile caractéristique des éléments dans gmsh.
    Info pour l'héritage : https://openclassrooms.com/fr/courses/235344-apprenez-a-programmer-en-python/233164-lheritage
    Tutorial : gmsh-4.0.2-Linux64/share/doc/gmsh/demos/api/t10.py
    """
    def __init__(self, tag=-1):
        self.in_model = False
        self.tag = None if tag==-1 else tag
        self.parents = None

    def set_params(self):
        """
        Cette partie sera précisée pour chaque type de champ de scalaire.
        """
        #? C'est a priori pas nécessaire de définir une méthode set_params dans cette classe.
        pass

    def add_gmsh(self):
        """
        Partie générique des intructions nécessaires pour ajouter un field de contrôle de la taille des éléments. 
        """

        if self.in_model:
            return self.tag #Le tag est là juste pour info
        if self.parents:
            for p_field in self.parents:
                if not pt.in_model:
                    p_field.add_gmsh()
        ipt_tag = self.tag if self.tag else -1
        api_tag = api_field.add(self.f_type, ipt_tag)
        self.set_params() #! Est-ce que c'est bon si l'on fait comme ça ?
        if not self.tag : 
            self.tag = api_tag
        else : 
            assert api_tag == self.tag
        self.in_model = True

    @staticmethod
    def set_background_mesh(fields, tag=-1):
    """
    Input : une liste de champs devant être utilisés pour imposer la taille caractéristiques des éléments.
    Only one background field can be given for the mesh generation. (gmsh reference manual 26/09/2018, 6.3.1 Specifying mesh element size)
    Toutes les contraintes sont ramenées à un unique champ scalaire en prenant, en chaque point, le minimum des valeurs imposées.
    """
    final_field = MinField(fields, tag)
    final_field.add_gmsh()
    api_field.setAsBackgroundMesh(final_field.tag)
    return final_field
    #? Rédaction terminée ! Mais fonction à tester
    #TODO : Tester !

class AttractorField(Field):
    """
    Field de type Attractor. Calcul la distance entre un point courant du domaine sur lequel le maillage doit être défini et les attracteurs qui peuvent être des Points, des Lines ou des Arcs.
    Paramètres : 
            points : liste d'instances de Point utilisés comme attracteurs;
            curves : liste d'instances de Line ou Arc utilisés comme attracteurs;
            nb_pts_discretization : Nb de points utilisés pour la discrétisation de chaque élément 1D de 'curves'.
    """

    def __init__(self, points=[], curves=[], nb_pts_discretization=10, tag=-1):
        Field.__init__(self, tag)
        self.f_type = "Attractor"
        self.points = points if points else None
        self.curves = curves if curves else None
        if curves:
            self.nb_pts_discret = nb_pts_discretization

    def set_params(self):
        if self.points:
            for pt in self.points:
                #? La méthode add_gmsh contient déjà un test logique pour s'assurer que le Point n'est pas déjà dans le modèle. Est-ce que c'est mieux d'appeler la méthode et de faire le test de la méthode ou de faire un test, puis d'appeler la méthode (qui contient un second test) ?
                if not pt.in_model:
                    pt.add_gmsh()
            api_field.setNumbers(self.tag, "NodesList", [pt.tag for pt in self.points])
        if self.curves:
            api_field.setNumber(self.tag, "NNodesByEdge", self.nb_pts_discret)
            for crv in self.curves:
                if not crv.in_model:
                    crv.add_gmsh()
            api_field.setNumbers(self.tag, "EdgesList", [crv.tag for crv in self.curves])
    #* Rédaction OK !
    #TODO : À tester !

class ThresholdField(Field):
    """
    Field de type Threshold. Décroissance affine ou en sigmoïde de la longueur caractéristique aux environs d'attracteurs.
    Paramètres :
        d_min, d_max : distances from entity that define the area where the element size is interpolated between lc_min and lc_max;
        lc_min : caracteristic element size inside balls of raduis d_min and centered at each attractor;
        lc_max : caracteristic element size outside balls of raduis d_max and centered at each attractor;
        attract_field : field that define the attractors that are used as centers for the d_min and d_max radii balls.
 
    F = LCMin if Field[IField] <= DistMin,
    F = LCMax if Field[IField] >= DistMax,
    F = interpolation between LcMin and LcMax if DistMin < Field[IField] < Dist-Max
    """

    def __init__(self, attract_field, d_min, d_max, lc_min, lc_max, sigmoid_interpol=False, tag=-1):
        Field.__init__(self, tag)
        self.f_type = "Threshold"
        self.d = (d_min, d_max)
        self.lc = (lc_min, lc_max)
        self.parents = [attract_field]
        self.sigmoid = sigmoid_interpol

    def set_params(self):
        # if not self.parents[0].in_model:
        #     self.parents[0].add_gmsh()
        api_field.setNumber(self.tag, "IField", self.parents[0].tag)
        api_field.setNumber(self.tag, "LcMin", lc[0])
        api_field.setNumber(self.tag, "LcMax", lc[1])
        api_field.setNumber(self.tag, "DistMin", d[0])
        api_field.setNumber(self.tag, "DistMax", d[1])
    #* Rédaction OK !
    #TODO : À tester !


class RestrictField(Field):
    """
    'Restrict the application of a field to a given list of geometrical points, curves, surfaces or volumes.'
    """

    def __init__(self, inpt_field, points=[], curves=[], surfaces=[], tag=-1):
        Field.__init__(self, tag)
        self.f_type = "Restrict"
        self.points = points if points else None
        self.curves = curves if curves else None
        self.surfaces = surfaces if surfaces else None
        self.parents =  [inpt_field]

    def set_params(self):
        api_field.setNumber(self.tag, "IField", self.parents[0].tag)
        if self.points:
            for pt in self.points:
                if not pt.in_model:
                    pt.add_gmsh()
            api_field.setNumbers(self.tag, "VerticesList", [pt.tag for crv in self.points])
        if self.curves:
            for crv in self.curves:
                if not crv.in_model:
                    crv.add_gmsh()
            api_field.setNumbers(self.tag, "EdgesList", [crv.tag for crv in self.curves])
        if self.surfaces:
            for srf in self.surfaces:
                if not srf.in_model:
                    srf.add_gmsh()
            api_field.setNumbers(self.tag, "FacesList", [srf.tag for srf in self.surfaces])
    #* Rédaction OK !
    #TODO : À tester !


class MathEvalField(Field):
    """
     Evaluate a mathematical expression.
    
    Des champs peuvent être utilisés dans l'expression. Dans ce cas, les faire apparaitre dans le paramètre inpt_fields. 
    """

    def __init__(self, formula_str, inpt_fields=[], tag=-1):
        Field.__init__(self, tag)
        self.f_type = "MathEval"
        self.parents = inpt_fields
        self.formula = formula_str
    
    def set_params(self):
        api_field.setString(self.tag, "F", formula)


class MinField(Field):
    """
    """

    def __init__(self, inpt_fields, tag=-1):
        Field.__init__(self, tag)
        self.f_type = "Min"
        self.parents=inpt_fields

    def set_params(self):
        model.mesh.field.setNumbers(self.tag, "FieldsList", [f.tag for f in self.parents])


#? Est-ce que emballer tout ça dans un gros objet facilite l'utilisation et/ou la compréhension ? Pas sûr... même plutôt tendance à penser le contraire
class LocalMeshRefinement(object):
    """
    Permet de raffiner le maillage autour de certains Points géométriques à l'aide d'objets de type Field (champs scalaires).
    Le champ à utiliser par la suite est contenu dans l'attribut major_field
    """

    def __init__(self, d_min, d_max, lc_min, lc_max, attractors={'points':[],'curves':[]}, nb_pts_discretization=10, sigmoid_interpol=False, restrict=False, restrict_domain={'points':[],'curves':[], 'surfaces':[]}):
        try:
            inpt_points = attractors['points']
        except KeyError:
            inpt_points = []
        try:
            inpt_curves = attractors['curves']
        except KeyError:
            inpt_curves = []
        self.attract_field = AttractorField(inpt_points, inpt_curves,nb_pts_discretization)
        self.threshold_field = ThresholdField(self.attract_field, d_min, d_max, lc_min, lc_max)
        if restrict:
            try:
                rstrc_points = attractors['points']
            except KeyError:
                rstrc_points = []
            try:
                rstrc_curves = attractors['curves']
            except KeyError:
                rstrc_curves = []
            try:
                rstrc_surf = attractors['surfaces']
            except KeyError:
                rstrc_surf = []
            self.restrict_field = RestrictField(self.threshold_field, rstrc_points,rstrc_curves, rstrc_surf)
            self.major_field =  self.restrict_field 
        else:
            self.major_field =  self.threshold_field 


        



# def set_background_mesh(size_constraints, tag=-1):
#     """
#     Input : une liste d'objets correspondant à des contraintes imposées sur la taille caractéristiques des éléments, contenant un attribut major_field
#     If tag is positive, assign the tag explcitly; otherwise a new tag is assigned automatically.

#     Only one background field can be given for the mesh generation. (gmsh reference manual 26/09/2018, 6.3.1 Specifying mesh element size)
#     Toutes les contraintes sont ramenées à un unique champ scalaire en prenant, en chaque point, le minimum des valeurs imposées.
#     """
#     final_tag = api_field.add("Min", tag)
#     field_tags = [c.major_field.tag for c in size_constraints]
#     api_field.setNumbers(final_tag, "FieldsList", field_tags)
#     api_field.setAsBackgroundMesh(final_tag)
#     #? Rédaction terminée ! Mais fonction à tester
#     #TODO : Tester !




