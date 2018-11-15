# -*- coding: utf-8 -*-
"""
Created on 09/10/2018
@author: baptiste

Définition de classes d'objets géométriques et de fonctions permettant de créer un modèle géométrique de RVE dans gmsh.

sources : 
    - https://deptinfo-ensip.univ-poitiers.fr/ENS/doku/doku.php/stu:python:pypoo

"""

import copy
import math
import operator
import os
import warnings
import logging

import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np

import gmsh

logger = logging.getLogger(__name__) #http://sametmax.com/ecrire-des-logs-en-python/
logger.setLevel(logging.DEBUG)

# nice shortcuts
model = gmsh.model
factory = model.occ

warnings.simplefilter("error") #? Doc: https://docs.python.org/3.6/library/warnings.html

#TODO : regarder ou mettre ces commandes
# gmsh.initialize() #? A mettre dans un fichier init ????
# #? Uitliser cet argument ? gmsh.initialize(sys.argv) cf script boolean.py
# gmsh.option.setNumber("General.Terminal", 1) #0 or 1 : print information on the terminal

#TODO : à revoir
GEOMETRY_KERNELS = ["Built-in", "OpenCASCADE"] #!
#? Obsolète car travail que avec openCASCADE ? Choix d'utiliser des fonctions de gmsh/model/occ
#TODO : à revoir
DEFAULT_LC = 1. #!
#? Obsolète ?

#TODO : docstring à faire
def unit_vect(v):
    """ Renvoie le vecteur normé. Nécessite un vecteur non nul"""
    return v / np.linalg.norm(v)

#TODO : docstring à faire
def angle_between(v1, v2, orient=True):
    """ Renvoie l'angle en radian, orienté ou non entre les deux vecteurs.
    Valeur comprise entre -pi (strictement) et pi.
    """

    v1u = unit_vect(v1)
    v2u = unit_vect(v2)
    if orient:
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.arctan2.html
        # EN 2D SEULEMENT
        angle = np.arctan2(v2u[1], v2u[0]) - np.arctan2(v1u[1], v1u[0])
        if angle <= -math.pi:
            angle += 2 * math.pi
        if angle > math.pi:
            angle -= 2 * math.pi
        return angle
    else:
        return np.arccos(np.clip(np.dot(v1u, v2u), -1.0, 1.0))

#TODO : docstring à faire
#TODO : Refactoring
def bissect(v1, v2):
    """Renvoie la direction de la bissectrice d'un angle défini par deux vecteurs, sous forme d'un vecteur unitaire.
        Seulement en 2D pour l'instant.
        """

    uv = list(map(unit_vect, [v1, v2]))
    e3 = np.array((0., 0., 1.))

    # if all(np.shape(x) == (2,) for x in uv): #* travail directement avec vecteurs de dimension 3.
    #TODO : voir si d'autres répercussions dans le code
        # if uv[0].shape == (2,) and uv[1].shape == (2,) :
        #        vu=[np.hstack((v,np.zeros((1,)))) for v in vu]
        # Complèter les vecteurs 2D par une troisième coordonnée nulle n'est pas nécessaire.
        # Fait automatiquement avec numpy.cross
    biss = np.cross(uv[1] - uv[0], e3)
    #biss = np.delete(biss, 2, 0)  # retour à un vecteur 2D. Maintenant on travaille en dimension 3
    biss = unit_vect(biss)
    return biss

def dual_base(basis):
    """
    Calculates the dual basis associated with a given basis. 

    Parameters
    ----------
    basis : numpy array_like, square matrix
        The components of the basis vectors in an orthogonal coordinate system.
        2-D array, Dimensions of the matrix : 2×2 or 3×3
    
    Return
    ------
    dual_b : np.array
        The components of the vectors that composed the dual base, in the same orthogonal coordinate system.
    
    """
    return np.linalg.inv(basis).T

#TODO : doctring à faire
#TODO : Utiliser une fonction de l'API gmsh ? 
def geometry_kernel(script, choix=1): #! Obsolète. Travail directement que avec des fonctions géométriques de la classe occ de l'API. 
    """ Choisir le noyau géométrique utilisé par gmsh.

        0 = Built-in
        1 = OpenCASCADE
        La commande doit être écrit en début de script.
        Pour les opérations booléennes, OpenCASCADE est nécessaire.
        OpenCASCADE permet aussi une construction géométrique top-bottom.

    """
    cmd_str = 'SetFactory("%s");\n' % GEOMETRY_KERNELS[choix]
    script.write(cmd_str)

def init_geo_tools():
    """
    The Gmsh Python API must be initialized before using any functions. 
    In addition the counters that are used to get the tags/incremental ID of the geometry objects are set to 1. 
    """
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)
    
    Point.all_pts_in_script = []
    # PhysicalEntity.count = 1
    # PhysicalEntity.tagDejaPris = set()
    

class Point(object):
    """Classe définissant un point caractérisée par :
    - ses coordonnées en 2D ou 3D ;
    - (à rajouter ?) la dimension de l'espace de travail;
    - (à rajouter ?) son identifiant unique;
    - (à rajouter ?) sa longueur caractéristique pour le maillage

    Cette classe possède un attribut de classe qui s'incrémente à chaque fois que l'on crée un Point
    """
# TODO faire docstring

    all_pts_in_script = []  #TODO : Doit être gardé ? # Utile pour imposer une certaine longueur caractéristique d'éléments en fin de script
    # https://www.toptal.com/python/python-class-attributes-an-overly-thorough-guide 
    # "We could even use this design pattern to track all existing instances of a given class, rather than just some associated data."
    all_instances = [] #! Refactoring nécessaire pour regrouper avec all_pts_in_script

    def __init__(self, coord=np.array((0., 0.)), mesh_size=0):
        """Constructeur de notre classe. Coordonnées importéées sous forme d'un np.array"""
        dim = coord.shape[0]
        self.coord = coord
        if dim == 2:
           self.coord = np.append(self.coord, [0.])
        #? Choix : on utilise toujours des coordonnés en 3D. Les points définis en 2D sont mis dans le plan z=0. 
        self.tag = None
        #* Nouveau ! Pour identifier les points "importants" autour des quels il faut raffiner le maillage
        self.fine_msh = False #TODO A définir et voir son utilisation...
        self.mesh_size = mesh_size
        Point.all_instances.append(self)

    # def __repr__(self):#? Je ne m'en sers pas ou vraiment rarement. On supprime ? 
    #     """Methode pour afficher les coordonnées du point."""
    #     return "pt %s : %s \n" % (self.tag, self.coord)

#? Obsolète si on utilise la fonction occ.removeAllDuplicate() ? A priori non.
    def __eq__(self, other):
        """
         Opérateur de comparaison == redéfini pour les objets de la classe Point
         Renvoie True ssi les coordonnées sont égales, sans prendre en compte la signature de l'objet.
        """
        if not isinstance(other, Point):
            return False
        if np.array_equal(self.coord, other.coord):
            return True
        elif np.allclose(self.coord, other.coord):
            return True
        else:
            return False
        #? Pour alléger le code, préférer une rédaction comme ça : ?
        #? if not isinstance(other, Point):
        #?     return False
        #? if np.array_equal(self.coord, other.coord):
        #?     return True
        #? elif np.allclose(self.coord, other.coord):
        #?     return True
        #? return False

    def __ne__(self, other): #? Est-ce utile ?
        return not self.__eq__(other)

    #TODO Adapter à la 3D avec matplotlib option 3D ?
    #? Passer la couleur la taille et tout le reste en kargs, puis passer les kargs en parametres du plot() ?
    def plot(self, color="red", size=5):
        plt.plot(self.coord[0], self.coord[1], marker='o', markersize=size, color=color)

    def add_gmsh(self):
        """
        """
        if self.tag: # That means that the geometrical entity has already been instantiated in the gmsh model.
            return self.tag #The tag is returned for information purposes only.
        self.tag = factory.addPoint(self.coord[0], self.coord[1], self.coord[2], self.mesh_size)
        # Point.all_pts_in_script.append(self) #?Utile ?

    def rotation(self, centre, angle):
        #? À Remplacer par rotate ? 
        #TODO : passer à la 3D
        #TODO : indiquer axis par coord point et direction et non par un point
        """Rotation du point autour d'un centre de type Point précisé en argument """
        c, s = math.cos(angle), math.sin(angle)
        R = np.array(((c, -s), (s, c)))
        self.coord = np.dot(R, self.coord - centre.coord) + centre.coord
    # def rotate(self, axis_pt_coord, axis_dir, angle): #* nouvelle méthode, calque de la staticmethod rotate de l'API
    #     factory.rotate([(0, self.tag)], axis_pt.coord[0], axis_pt.coord[1], axis_pt.coord[2],axis_dir[0], axis_dir[1], axis_dir[2], angle)
    #     new_coord = None #! Problème car on a modifié les coordonnées du point dans le modèle gmsh
    #     # factory.synchronize()
    #     # new_coord = model.getValue(0,self.tag,[]) #! Opération couteuse ? Est-ce que ça fonctionne avant l'opération synchronize() ?
    #     # print("Nouvelles coordonnées", new_coord)

#* Pas utilisé pour le moment. Associer à la fonction dilate de l'API.
    # def homothetie(self, centre, k):
    #     """homothétie de rapport k du point par rapport à un centre de type Point précisé en argument """
    #     self.coord = centre.coord + k * (self.coord - centre.coord)

#* Pas utilisé pour le moment
    # def distance(self, other):
    #     """Return the distance between this point and a reference point"""
    #     return np.linalg.norm(self.coord - other.coord)



#### Fonctions permettant des opérations géométriques de bases sur des objets de type Point ####
def centroSym(pt, center):
    warnings.warn("Deprecated. Should use the point_reflection function instead.", DeprecationWarning)
    """ Renvoie un nouveau point obtenu par symétrie centrale."""
    new_coord = -(pt.coord - center.coord) + center.coord
    return Point(new_coord)

def mirrorSym(pt, centre, axe):
    """
    Renvoie un nouveau point obtenu par symétrie mirroir. axe de symétrie décrit par un point et un vecteur
    SEULEMENT EN 2D POUR LE MOMENT
    To do : généraliser 2D et 3D
    """
    warnings.warn("Deprecated. Should use the plane_reflection function instead.", DeprecationWarning)
    new_coord = (2 * (pt.coord - centre.coord).dot(axe) * axe - (pt.coord - centre.coord) + centre.coord)
    return Point(new_coord)

def translat(pt, vect):
    """ DEPRECATED. Translation d'un point, défini par un vecteur de type np.array
    """
    warnings.warn("Deprecated. Should use the translation function instead.", DeprecationWarning)
    new_coord = pt.coord + vect
    return Point(new_coord)

def offset(pt, pt_dir1, pt_dir2, t):
    v1 = pt_dir1.coord - pt.coord
    v2 = pt_dir2.coord - pt.coord
    alpha = angle_between(v1, v2, orient=False)
    v_biss = bissect(v1, v2)
    if alpha != np.pi:
        dpcmt = -t / abs(np.sin(alpha / 2.))
    else:
        dpcmt = -t
    new_coord = pt.coord + dpcmt * v_biss
    return Point(new_coord)

class Curve(object):
    """
    Superclass that is used to define both the Line and the Arc classes.
    It is designed to represent geometrical entities of dimension one.

    """
    all_instances = []
    def __init__(self, def_pts_list):
        self.def_pts = def_pts_list
        self.tag = None
        self.gmsh_constructor = None
        Curve.all_instances.append(self)
    
    def __eq__(self, other):
        """
        Return True if and only if : 
        - both self and other are instances of the same subclass,
        AND
        - The coordinates of the points that are used to define these two Line (or Arc) are equal.

        """
        if not type(other) is type(self):
            return False
        return all(p == q for p, q in zip(self.def_pts, other.def_pts))

    def __ne__(self, other):
        return not self.__eq__(other)

    def add_gmsh(self):
        #TODO : refaire docstring
        """Méthode permettant d'ajouter une ligne de script dans un fichier .geo pour créer la ligne,
        et les deux points extrémités si nécessaire.
        """
        if self.tag: # Geometrical entity has already been instantiated in the gmsh model.
            return self.tag # For information purposes only.
        for pt in self.def_pts:
            if not pt.tag:
                pt.add_gmsh()
        self.tag = self.gmsh_constructor(*[p.tag for p in self.def_pts])
    
    def get_tag(self):
        warnings.warn("Deprecated. Should use explicit if not self.tag checks before reading set.tag instead.", DeprecationWarning)
        pass
        # if self.tag:
        #     return self.tag
        # else:
        #     self.add_gmsh()
        #     return self.tag
        # #? Autre idée : @property
        if self.tag:
            self.tag *= -1

class Line(Curve):
    """Classe définissant une ligne simple caractérisée par :
    - son point de départ
    - son point d'arrivée
    """
    #TODO : Créer une méthode pour inverser le sens de la ligne ?

    def __init__(self, start_pt, end_pt):
        Curve.__init__(self, [start_pt, end_pt])
        self.gmsh_constructor = factory.addLine

    def __str__(self):
        """Affichage plus clair des coordonnées des points de départ et d'arrivée."""
        prt_str = "Line " + self.tag if self.tag else "--" + ", "
        prt_str += "start point tag : %i , end point tag : %i" %(self.def_pts[0].tag, self.def_pts[1].tag)
        return prt_str

    def longueur(self):
        return np.linalg.norm(self.def_pts[0].coord - self.def_pts[1].coord)

    def direct(self):
        """ Renvoie un vecteur unitaire correspondant à la direction de la ligne"""
        return 1 / self.longueur() * (self.def_pts[1].coord - self.def_pts[0].coord)

    def plot(self, color="black"):
        """En 2D seulement. Tracer la ligne dans un plot matplotlib. """
        x = [pt.coord[0] for pt in self.def_pts]
        y = [pt.coord[1] for pt in self.def_pts]
        plt.plot(x, y, color=color)


class Arc(Curve):
    """Classe définissant un arc de cercle caractérisé par :
    - son point de départ
    - son point d'arrivée
    - son centre
    - son rayon
    """

    def __init__(self, start_pt, center_pt, end_pt):
        """ Crée un arc de cercle, après avoir comparé les distances entre le centre et les deux extrémités indiquées.
         https://docs.scipy.org/doc/numpy/reference/generated/numpy.testing.assert_array_almost_equal.html
        """

        d1 = np.linalg.norm(start_pt.coord - center_pt.coord)
        d2 = np.linalg.norm(end_pt.coord - center_pt.coord)
        np.testing.assert_almost_equal(d1, d2, decimal=10)
        Curve.__init__(self, [start_pt, center_pt, end_pt])
        self.gmsh_constructor = factory.addCircleArc
        self.radius = (d1 + d2) / 2

    def __str__(self):
        # prt_str = "Arc %s \n Pt début : %s Pt fin : %s Centre : %s" %(self.name, str(self.deb), str(self.fin), str(self.centre))
        prt_str = "Circle arc " + self.tag if self.tag else "--" + ", "
        prt_str += "start point tag : %i , center tag : %i , end point tag : %i" %(self.def_pts[0].tag, self.def_pts[1].tag, self.def_pts[2].tag)
        return prt_str

    def plot(self, circle_color="Green", end_pts_color="Blue", center_color="Orange", pt_size=5):
        """Représenter l'arc de cercle dans un plot matplotlib. 
        Disponible seulement en 2D pour l'instant."""

        self.def_pts[0].plot(end_pts_color, pt_size)
        self.def_pts[2].plot(end_pts_color, pt_size)
        self.def_pts[1].plot(center_color, pt_size)
        circle = plt.Circle((self.def_pts[1].coord[0],self.def_pts[1].coord[1]), self.radius, color=circle_color, fill=False)
        ax = plt.gca()
        ax.add_patch(circle)

#! Replaced by the add_gmsh method of the base class curve.
    # def add_gmsh(self):
    #     """Méthode permettant d'ajouter une ligne de script dans un fichier .geo pour l'arc de cercle,
    #     et si nécesaire les deux points extrémités et le centre.
    #     """
    #     if self.tag: # Geometrical entity has already been instantiated in the gmsh model.
    #         return self.tag # For information purposes only.

    #     for pt in self.def_pts:
    #         if not pt.tag:
    #             pt.add_gmsh()
    #     self.tag = factory.addCircleArc(self.def_pts[0].tag, self.def_pts[1].tag, self.def_pts[2].tag)


class AbstractCurve(Curve):
    """
    """
    def __init__(self, tag):
        """
        Créer une représentation d'une courbe existant dans le modèle Gmsh.

        #! A corriger Lors de l'instantiation, les points extrémités peuvent être donnés, soient explicitement, soit par leurs tags.

        """
        Curve.__init__(self, [])
        self.tag = tag
    
    def get_boundary(self, get_coords=True):
        """
        Récupérer les points correspondants aux extrémités de la courbe à partir du modèle Gmsh.

        Parameters
        ----------
        coords : bool, optional
            If true, les coordonnées des points extrémités sont aussi récupérés.
        
        """
        def_pts = []
        #! Pour debug print ("tag in get_boundary method of AbstractCurve", self.tag)
        boundary = model.getBoundary((1, self.tag), False, False, False)
        # print("getBoundary results, for AbstractCurve %i : "%self.tag, boundary)
        for pt_dimtag in boundary:
            if not pt_dimtag[0] == 0:
                raise TypeError("The boundary of the geometrical entity %i are not points." %self.tag)
            #! BUG for pt in Point.all_instances:
            for pt in []: #? TEST
                if pt.tag == pt_dimtag[1]:
                    print("One end of the AbstractCurve instance is already represented by an instance of Point!")
                    def_pts.append(pt)
                    break
            else:
                #! getValue call requires the model to be synchronized !
                coords = model.getValue(0, pt_dimtag[1], []) if get_coords else []
                logger.debug(repr(coords))
                new_pt = Point(np.array(coords))
                new_pt.tag = pt_dimtag[1]
                def_pts.append(new_pt)
        self.def_pts = def_pts

    def plot(self, color="black"):
        """En 2D seulement. Tracer les points de def_pts reliés dans un plot matplotlib. """
        x = [pt.coord[0] for pt in self.def_pts]
        y = [pt.coord[1] for pt in self.def_pts]
        plt.plot(x, y, color=color, linestyle = 'dashed')

class LineLoop(object):
    """
    Définit une courbe fermée, composée de segments et d'arc.
    """

    def __init__(self, elements, explicit=False):
        """
        La LineLoop est soit défini par des sommets, soit directement défini par une liste d'objets Line/Arcs (explicit = True)
        """

        # self.log = list()  # Savoir quelles opérations ont déjà été faites ?
        self.info_offset = False #! A remplacer par quelque chose de mieux. Comme l'utilisation de l'attribut "vertices"
        if explicit:
            self.sides = elements
        else:
            self.vertices = elements
            self.sides = list()
        self.tag = None

    def __eq__(self, other):
        """
         Opérateur de comparaison == surchargé pour les objets de la classe LineLoop
         Si la LineLoop n'est définie que par ses sommets, renvoie True ssi les listes de sommets sont égales, à un décalage d'indice près.
         Si la LineLoop est aussi définie par des Line/Arc, renvoie True ssi
         l'ensemble des éléments 1D qui composent la LineLoop est identique à celui de la LineLoop comparée.
         L'orientation est prise en compte.

         """
        if not isinstance(other, LineLoop):
            return False
        if self.sides or other.sides:  # Si l'une des deux LineLoop est définie par des Line/Arc alors on compare au niveau de ces éléments.
            if len(self.sides) != len(other.sides):
                return False
            else:
                for elmt_1D in self.sides:
                    for other_elmt in other.sides:
                        test = elmt_1D == other_elmt
                        if test:
                            break
                    else:  # Si aucun break n'est déclenché. C'est à dire si l'un des cote de la lineloop courante n'appartient pas à la LineLoop comparée
                        return False
                else:
                    return True
        if len(self.vertices) != len(other.vertices):
            return False
        else:
            for shift in range(len(self.vertices)):
                if all(p == other.vertices[i - shift] for i, p in enumerate(self.vertices)):
                    return True
            else:
                return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def plot(self, color="black"):
        """Représenter la polyligne dans un plot matplotlib. 
        Disponible seulement en 2D pour l'instant."""
        if not self.sides:
            self.vertices_2_sides()
        for elmt in self.sides:
            elmt.plot(color)

    def add_gmsh(self):
        if self.tag: # Geometrical entity has already been instantiated in the gmsh model.
            return self.tag # For information purposes only.
        if not self.sides:
            self.vertices_2_sides()
        for elmt in self.sides:
            if not elmt.tag:
                elmt.add_gmsh()
        self.tag = factory.addCurveLoop([elmt.tag for elmt in self.sides])

    def reverse(self):
        self.sides.reverse()
        for elmt in self.sides:
            elmt.def_pts[0], elmt.def_pts[-1] = elmt.def_pts[-1], elmt.def_pts[0]

    def offset(self, t):
        """ Opération d'offset appliqué automatique sur tout les sommets définissant la LineLoop.
        Cette opération doit donc être faite assez tôt, avant que les Line/Arc composant la LineLoop soit créés.
        """
        assert not self.sides  # Si on est déjà en présence de Lines, il est trop tard pour faire l'offset de cette façon
        new_vrtces = [None]*(len(self.vertices))
        self.info_offset = True
        for i in range(len(self.vertices)):
            new_vrtces[i - 1] = offset(self.vertices[i - 1], self.vertices[i - 2], self.vertices[i], t)
        self.offset_dpcmt = [np.linalg.norm(new.coord - prev.coord) for new, prev in zip(new_vrtces, self.vertices)] #TODO  : regarder où c'est utiliser et si on peut avoir quelque chose de plus clair
        self.vertices = new_vrtces

    def round_corner_explicit(self, radii):
        """Opération d'arrondi des angles appliquée à tous les sommets du polygone.
        Les rayons sont indiqués de manière explicite, sous forme d'une liste. liste de longueur 1 pour un rayon uniforme.
        """
        if  isinstance(radii, list):
            radii = [radii[i % len(radii)] for i in range(len(self.vertices))]
        else:
            radii = [radii]*(len(self.vertices))
        result_1D = list()
        for i in range(len(self.vertices)):
            result_1D.append(round_corner(self.vertices[i - 1], self.vertices[i - 2],
                                          self.vertices[i], radii[i - 1], False, False))
        self.round_corner_2_sides(result_1D)


    def round_corner_incircle(self, radii):
        """ Opération d'arrondi des angles appliquée à tous les sommets du polygone.
        La méthode du cercle inscrit est utilisée.
        radii = liste de rayons à utiliser ou valeur (float) si rayon uniforme.
        Une liste de rayons de cercles inscrits peut être indiquée, liste de longueur 1 pour un rayon uniforme.
        Dans le cas où la longueur de la liste de rayon est ni 1 ni égale au nombre de sommets, un modulo est utilisé.
        """
        if  isinstance(radii, list):
            effect_R = [radii[i % len(radii)] for i in range(len(self.vertices))]
        else:
            effect_R = [radii]*(len(self.vertices))
        if self.info_offset:
            effect_R = map(operator.sub, effect_R, self.offset_dpcmt)
        result_1D = list()
        for i in range(len(self.vertices)):
            result_1D.append(round_corner(self.vertices[i - 1], self.vertices[i - 2],
                                          self.vertices[i], effect_R[i - 1], True, False))
        self.round_corner_2_sides(result_1D)


    def round_corner_2_sides(self, result_list): #? Est-ce vraiment sa place dans l'architecture du code ?
        """ Permet de traiter les résultats d'une opération round_corner appliquée en série sur un ensemble de sommets.
        Une polyligne composée de segments et d'arc est composée puis stockée dans l'attribut sides.
        """

        for i, rslt in enumerate(result_list):
            new_line = rslt[0]
            new_line.def_pts[0] = result_list[i - 1][1].def_pts[-1]  # Correction pour que le segment commence à la fin de l'arc précédent.
            new_arc = rslt[1]
            self.sides.extend([new_line, new_arc])

    def vertices_2_sides(self):
        """ Méthode permettant de générer automatiquement les segments reliant les sommets, stockés dans l'argument sides.
        Si une opération round_corner est utilisé, cette opération est inutile."""
        if self.sides:
            print("Warning : attribut sides d'une LineLoop écrasé lors de l'utilisation de la méthode vertices_2_sides.")
        self.sides = [Line(self.vertices[i - 1], self.vertices[i]) for i in range(len(self.vertices))]


def centro_sym_ll(inp_ll, center):
    warnings.warn("Deprecated. Should use the point_reflection function instead.", DeprecationWarning)
    """Opération de symétrie centrale sur une LineLoop ou liste de LineLoop. Une nouvelle LineLoop ou liste de LineLoop est renvoyée.
    La symétrie est faite aux niveaux des sommets (objets Points) de la LineLoop et n'agit pas sur l'attribut sides.
    """
    if type(inp_ll) is not list:
        new_pts = [centroSym(pt, center) for pt in inp_ll.vertices]
        return LineLoop(new_pts, explicit=False)
    else:
        new_ll_list = [centro_sym_ll(each_ll, center) for each_ll in inp_ll]
        return new_ll_list


def mirror_sym_ll(inp_ll, center, axe):
    
    """Une nouvelle LineLoop ou liste de LineLoop est calculée par symmétrie mirroir et renvoyée.
    La symétrie est faite aux niveaux des sommets (objets Points) de la LineLoop et n'agit pas sur l'attribut sides.
    Le sens de rotation de la LineLoop (horaire / anti_horaire) est conservé.
    """
    warnings.warn("Deprecated. Should use the point_reflection function instead.", DeprecationWarning)
    if type(inp_ll) is not list:
        new_pts = [mirrorSym(pt, center, axe) for pt in inp_ll.vertices]
        new_pts.reverse()
        return LineLoop(new_pts, explicit=False)
    else:
        new_ll_list = [mirror_sym_ll(each_ll, center, axe) for each_ll in inp_ll]
        return new_ll_list


def translat_ll(inp_ll, vect):
    """ DEPRECATED.  Doc_string à compléter. Voir mirror_sym_ll pour un fonctionnement similaire. """
    
    warnings.warn("Deprecated. Should use the translation function instead.", DeprecationWarning)
    if type(inp_ll) is not list:
        new_pts = [translat(pt, vect) for pt in inp_ll.vertices]
        return LineLoop(new_pts, explicit=False)
    else:
        new_ll_list = [translat_ll(each_ll, vect) for each_ll in inp_ll]
        return new_ll_list

#! #! #!
#TODO : Pour la classe LineLoop : 
#TODO	Utiliser des attributs Sommets_de_base, et côtés pour faire en sorte que les sommets soient reconstruit si on crée la LineLoop de manière explicit
#?  	Faire 2 constructors au lieu du paramètre Explicit ?
#TODO garder une info "en dur" sur la position des sommets du polygone correspondant à la LineLoop plutôt que d'utiliser une booléen pour l'historique de l'offset et un argument qui garde en mémoire les déplacements ?

class PlaneSurface(object):
    """
    Calque de la fonction Plane Surface native de gmsh
    Créée à partir d'une LineLoop définissant le contour extérieur et, si nécessaire, de line loops définissant des trous internes
    """

    def __init__(self, ext_contour, holes=[]):
        self.ext_contour = ext_contour
        self.holes = holes
        self.tag = None
    
    def __eq__(self, other):
        """
         Opérateur de comparaison == surchargé pour les objets de la classe Plane Surface
         Orientation prise en compte. Considérer le cas de surfaces non orientées ????
        """

        if not isinstance(other, PlaneSurface):
            return False
        if self.ext_contour != other.ext_contour:
            return False
        if len(self.holes) != len(other.holes):
            return False
        if len(self.holes) != 0:
            for contour in self.holes:
                for unContourOther in other.holes:
                    if contour == unContourOther:
                        break
                else:
                    return False
        return True

    def add_gmsh(self):
        if self.tag:
            return self.tag
        all_loops = [self.ext_contour] if not self.holes else [self.ext_contour]+self.holes
        for ll in all_loops:
            if not ll.tag:
                ll.add_gmsh()
        self.tag = factory.addPlaneSurface([ll.tag for ll in all_loops])

class AbstractSurface(object):
    """
    Surface dont on ne connait rien à part le tag.
    Une instance d'AbstractSurface existe dans le model gmsh et est récupérée grâce à l'API.
    Il s'agit par exemple du résulat d'une opération booléenne.
    Par contre, ses bords sont a priori inconnus
    """

    def __init__(self, tag):
        self.tag = tag
        self.boundary = []
    
    def get_boundary(self, recursive=True):
        """
        Récupérer les tags des entitées géométriques 1D qui composent le bord de la surface.
        
        Parameters
        ----------
        recursive : bool, optional
        If True, the boundaries of the 1-D entities that form the boundary of the AbstractSurface instance are also extracted from the gmsh model. 
        #? Utile ? Instances of Point are created to represent them.

        """
        #! Pour debug print(self.__dict__)
        def_crv = []
        boundary = model.getBoundary((2, self.tag), False, False, False)
        # if isinstance(self.tag, list):
        #     boundary = model.getBoundary([(2, t) for t in self.tag], combine=True, oriented=False, recursive=False)
        # else:
        #     boundary = model.getBoundary([(2, self.tag)], False, False, False)
        logger.debug(repr(boundary))
        for crv in boundary:
            if not crv[0] == 1:
                print("[Warning] Unexpected type of geometrical entity in the boundary of surface %i" %self.tag)
                continue
            for instc in [] : #! Curve.all_instances: TENTATIVE
                if instc.tag == crv[1] :
                    def_crv.append(instc)
                    break
            else:
                def_crv.append(AbstractCurve(crv[1]))
        
        if recursive:
            for crv in def_crv:
                if isinstance(crv, AbstractCurve):
                    crv.get_boundary()
        
        self.boundary = def_crv


#TODO : Not tested yet
# def combine_AbstractSurface(surfs):
#     """
#     Combine several AbstractSurface into one. This operation may be useful before request for the boundary of a whole RVE.

#     Parameters
#     ----------
#     surfs : list of instances of AbstractSurface or PlaneSurface
#     """
#     for s in surfs :
#         if not s.tag: #May occure if s is there are PlaneSurface instances in the input list.
#             s.add_gmsh()
#     combined_s = AbstractSurface([s.tag for s in surfs])




# def bool_cut(body, tools, tag=-1):
# """
# Remove the tool entities from the body entity. Removing a set of geometrical entities 'tools' is possible.
# Si plusieurs tools : l'input doit être une liste
# option pour supprimer ou non les tools. 
# """
#     pass


def bool_cut_S(body, tool):
    """
    Boolean operation of cutting performed on surfaces.

    Remove the aeras taken by the tool entities from the body surface.
    Removing a set of geometrical entities 'tools' at once is possible.
    The removeObject and removeTool parameters of the gmsh API function are set to False in order to keep the consistency between the python geometrical instances and the gmsh geometrical model as far as possible.

    Parameters
    ----------
    body : instance of PlaneSurface
        Main operand of the cut operation.
    tool : instance of PlaneSurface or list of instances of it
        Several tool areas can be removed to the body surface at once. To do this, the tool parameter must be a list.
    NOT ENABLED ANYMORE remove_body : bool, optional
        Delete the body surface from the gmsh model after the boolean operation.
        If True, the tag of the resulting surface might be equal to the one of the deleted body.
    NOT ENABLED ANYMORE remove_tool : bool, optional
        Delete the tool surface from the gmsh model after the boolean operation, or all the tools if several tools are used.
    
    Return
    ----------
    cut_surf : Instance of PlaneSurface
        A Python object that represents the surface that is obtained with the boolean operation.
        This will be a degenerate instance with only a tag attribut and a boundary attribut that can be evaluate later.
    """
    if not body.tag:
        body.add_gmsh()
    if isinstance(tool, PlaneSurface):
        tool = [tool]
    assert isinstance(tool, list)
    for t in tool:
        if not t.tag:
            t.add_gmsh()
    output = factory.cut([(2,body.tag)], [(2,t.tag) for t in tool], removeObject=False, removeTool=False)
    logger.debug(f"Output of boolean operation 'cut' on surfaces : {output}")
    new_surf = list()
    for entity in output[0]:
        if entity[0]==2:
            new_surf.append(AbstractSurface(entity[1]))
        else:
            logger.warn(f"Some entities that result from a cut boolean operation are not surfaces and therefore are not returned. \n Complete output from the API function :{output}")
    return new_surf

def bool_intersect_S(body, tool):
    """
    Boolean operation of intersection performed on surfaces.

    See the bool_cut_S() doc for more informations.

    """
    if not body.tag:
        body.add_gmsh()
    if isinstance(tool, PlaneSurface):
        tool = [tool]
    assert isinstance(tool, list)
    for t in tool:
        if not t.tag:
            t.add_gmsh()
    output = factory.intersect([(2,body.tag)], [(2,t.tag) for t in tool], removeObject=False, removeTool=False)
    logger.debug(f"Output of boolean operation 'intersection' on surfaces : {output}")
    new_surf = list()
    for entity in output[0]:
        if entity[0]==2:
            new_surf.append(AbstractSurface(entity[1]))
        else:
            logger.warn(f"Some entities that result from a intersection boolean operation are not surfaces and therefore are not returned. \n Complete output from the API function :{output}")
    return new_surf


def gather_boundary_fragments(curves, main_crv):
    """
    DEPRECATED
    Extract from a set of curves (1-D geometrical entities) those which represent a part of the main curve.

    Parameters
    ----------
    curves: list of instances of a Curve subclass
    main_crv : instance of Curve subclass

    """
    warnings.warn("Deprecated. Should use the macro_line_fragments function instead.", DeprecationWarning)
    if not main_crv.tag:
        main_crv.add_gmsh()
    for c in curves:
        if not c.tag:
            c.add_gmsh()
            print("[Info] In gather_boundary_fragments, curve {} added to the model".format(c.tag))
    parts = []
    for c in curves:
        factory.synchronize()
        # print ("main curve tag : {}, part candidate tag : {}".format(main_crv.tag,c.tag))
        output = factory.intersect([(1,main_crv.tag)], factory.copy([(1,c.tag)]), removeObject=False, removeTool=True) #! Methode détournée qui fonctionne. Dans le test élémentaire, si on prend removeTool=False, avec les courbes à trier, toutes ne sont pas détectées.
        # print("dans gather boundary, pour la courbe %i, l'output est : "%c.tag , output)
        output_1D = [dimtag[1] for dimtag in output[0] if dimtag[0] == 1]
        if output_1D:
            parts.append(c)
    return parts

def macro_line_fragments(curves, main_line):
    """
    Extract from a set of curves (1-D geometrical entities) those which represent a part of the main line.

    Designed to identify in the list of the boundary elements of the microstructure
    those that are on a given border of the RVE.

    Parameters
    ----------
    curves: list of instances of a Curve subclass
    main_line : instance of Curve subclass

    """
    for ln in curves + [main_line]:
        if not ln.tag:
            ln.add_gmsh()
        if not ln.def_pts:
            ln.get_boundary()
            logger.debug(f"In gather_boundary_fragments, curve {ln.tag} added to the model")
    main_start = main_line.def_pts[0].coord
    main_dir = main_line.def_pts[-1].coord - main_line.def_pts[0].coord
    parts = list()
    for crv in curves:
        crv_dir = crv.def_pts[-1].coord - crv.def_pts[0].coord
        if not np.allclose([0,0,0], np.cross(main_dir, crv_dir), rtol=0, atol=1e-08):
            continue
        mix_dir = 1/2*(crv.def_pts[-1].coord + crv.def_pts[0].coord) - main_start
        if not np.allclose([0,0,0], np.cross(main_dir, mix_dir), rtol=0, atol=1e-08):
            continue
        parts.append(crv)
    return parts

#TODO : à mieux définir
    # def delGmsh(self, script, recursive=False):
    #     delGmsh(self, "Surface", script, recursive)

#TODO : à déplacer dans mesh ?
class PhysicalGroup(object):
    """
    Create and manage groups of geometrical entities in the gmsh model.
    Physical groups can be visible in the exported mesh.

    "Groups of elementary geometrical entities can also be defined and are called “physical”
    groups. These physical groups cannot be modified by geometry commands: their only
    purpose is to assemble elementary entities into larger groups so that they can be referred to
    by the mesh module as single entities." (gmsh reference manual 26/09/2018)

    """
    def __init__(self, entities, geo_dim, name=None):
        """
        Gather multiple instances of one of the geometrical Python classes (Point, Curve, PlaneSurface) to form a single object in the gmsh model.
        
        Parameters
        ----------
        entities : list
            The instances that will compose the physical group.
            They must have the same geometrical dimension (0 for points, 1 for line, arcs, instances of AbstractCurve, 2 for surfaces, 3 for volumes)
        geo_dim : int
            Geometrical dimension of the entities that are gathered.
        name : string, optional
            name of the group.

        """
        self.entities = entities
        self.dim = geo_dim
        self.name = name
        self.tag = None

    def add_gmsh(self):
        if self.tag:
            return self.tag
        tags = list()
        print (self.entities)
        for item in self.entities:
            if not item.tag:
                item.add_gmsh()
            tags.append(item.tag)
        self.tag = model.addPhysicalGroup(self.dim, tags)
        if self.name:
            model.setPhysicalName(self.dim, self.tag, self.name)

    def add_to_group(self, entities):
        """
        Add a geometrical entity or a list of geometrical entities to an existing physical group.

        The appended items must be of the same geometrical dimension.
        """
        if self.tag:
            raise AttributeError("The physical group has been already defined in the gmsh model. It is too late to add entities to this group.")
        if isinstance(entities, list):
            self.entities += entities
        else :
            self.entities.append(entities)

    def set_color(self, rgba_color, recursive=False):
        """
        Choisir la couleur du maillage qui coincidera avec les éléments géométriques contenus de l'entité physique.
        
       Parameters
       ----------
        rgba_color : list of 4 integers between 0 and 255.
            RGBA code of the desired color.
        recursive : bool, optional
            Apply the color setting to the parent geometrical entities as well.
        """
        dimtags = [(self.dim, e.tag) for e in self.entities]
        model.setVisibility(dimtags, 1)
        model.setColor(dimtags, *rgba_color, recursive=recursive)


#TODO : Faire un choix.
#? Est ce qu'on met plutôt ce remove sous forme d'une méthode dans chaque classe d'objet géométrique ?
def remove_gmsh(geo_entity, recursive=False):
    """
    """
    #TODO : Faire docstring
    if not geo_entity.in_model:
        return False #? Renvoyer False car aucune opération n'est faite ? Autre output plus pertinent dans le cas où rien n'est fait ? Ou aucun output ?
    factory.remove([(geo_entity.geo_dim, geo_entity.tag)], recursive=recursive)
    geo_entity.in_model = False
    geo_entity.tag = None #! Qu'est ce qu'on fait du tag ?
    if not recursive:
        return True
    if geo_entity.geo_dim == 1: #Si on a affaire à des instances de Line or Arc
        for pt in geo_entity.def_pts:
            pt.in_model = False
            pt.tag = None #! Qu'est ce qu'on fait du tag ?
    # if geo_entity.geo_dim == 2: #? Bizarrre de s'interessé à geo_dim pour des instances de Surface alors que c'est la seule classe qui correspond à cette valeur de geo_dim. Utiliser geo_dim (pour avoir qqchose d'homogène) ou plutôt isinstance ?
    #     geo_entity.ext_contour.in_model = False
    #     geo_entity.ext_contour.tag = None  #! Qu'est ce qu'on fait du tag ?
    #     for entity_1D in geo_entity.ext_contour :
    #     if geo_entity.holes:  #* None est équivalent à False
    #TODO : TO BE CONTINUED....
    #! Problème à gérer !!!! Si un parent est aussi un parent d'une autre entité géométrique, le recursive remove ne va probablement pas le supprimer !!!!
    #! REPRENDRE ICI !
    #? Pour test si un attribut existe https://stackoverflow.com/questions/610883/how-to-know-if-an-object-has-an-attribute-in-python/610923#610923

# def delGmsh(objet, typeObjet, script, recursive=False):
#     """
#     écrire dans le script gmsh une ligne de commande permettant de supprimer un objet gmsh de type
#     """

#     outputline = "Delete { " + typeObjet + "{" + str(objet.idx) + "}; } \n"
#     if recursive:
#         outputline = "Recursive " + outputline
#     script.write(outputline)


# def suppr_doublon(inpList):
#     """
#     Supprimer les doublons dans une liste, en conservant l'ordre.
#     Fonction conçue pour fonctionner avec des listes de LineLoop.
#     """
#     newList = list()

#     for elmt in inpList:
#         for dejaPresent in newList:
#             if elmt == dejaPresent:
#                 # print "element suppprimé", elmt
#                 # POUR VOIR CE QUE L'ON SUPPRIME
#                 # fig=plt.figure()
#                 # ax=fig.add_subplot(1,1,1)
#                 # elmt.plot()
#                 # plt.axis('equal')
#                 # plt.show()
#                 break
#         else:
#             newList.append(elmt)
#     # Faire avec : [nv.append(item) for item in liste if not item in nv] ? comment le test d'appartenance fonctionne avec des classes perso ? A TESTER
#     return newList


def round_corner(inp_pt, pt_amt, pt_avl, r, junction_raduis=False, plot=False):
    """
    Calcul un arc de cercle nécessaire pour arrondir un angle défini par trois points.

    Le rayon de l'arc peut être contrôlé explicitement ou par la méthode du cercle inscrit.
    Deux objets de type Line sont aussi renvoyés.
    Ces segments orientés relient les deux points en amont et aval de l'angle à l'arc.

    """
    # Direction en amont et en aval
    v_amt = unit_vect(pt_amt.coord - inp_pt.coord)
    v_avl = unit_vect(pt_avl.coord - inp_pt.coord)

    alpha = angle_between(v_amt, v_avl, orient=True)
    v_biss = bissect(v_amt, v_avl)
    if alpha < 0:  # corriger le cas des angles \in ]-pi;0[ = ]pi;2pi[]. L'arrondi est toujours dans le secteur <180°
        v_biss = -v_biss

    if junction_raduis:
        if plot: R = copy.deepcopy(r)
        r = r * abs(math.tan(alpha / 2.)) * abs(math.tan(abs(alpha / 4.) + math.pi / 4.))

    # Calcul des distances centre - sommet de l'angle et sommet - point de raccordement
    dist_center = float(r) / abs(math.sin(alpha / 2.))
    dist_racc = float(r) / abs(math.tan(alpha / 2.))

    pt_racc_amt = translation(inp_pt, dist_racc * v_amt)
    pt_racc_avl = translation(inp_pt, dist_racc * v_avl)
    pt_ctr = translation(inp_pt, dist_center * v_biss)

    round_arc = Arc(pt_racc_amt, pt_ctr, pt_racc_avl)
    racc_amt = Line(pt_amt, pt_racc_amt)
    racc_avl = Line(pt_racc_avl, pt_avl)
    geoList = [racc_amt, round_arc, racc_avl]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        if junction_raduis:
            ax.add_patch(plt.Circle(inp_pt.coord, R, color="orange", fill=True))
        pts = [inp_pt, pt_amt, pt_avl]
        colors = ["black", "red", "blue"]
        x, y = list(), list()
        for pt, color in zip(pts, colors):
            pt.plot(color)
            x.append(pt.coord[0])
            y.append(pt.coord[1])
        pt_racc_amt.plot("black")
        pt_racc_avl.plot("black")
        pt_ctr.plot("purple")
        racc_amt.plot("red")
        racc_avl.plot("blue")
        ax.add_patch(round_arc.plot("purple"))
        plt.axis('equal')
        plt.xlim(min(x), max(x))
        plt.ylim(min(y), max(y))
        plt.show()

    return geoList


def remove_duplicates(ent_list): #! Fonctionne très bien aussi pour d'autres types d'objets géométriques (Points par exemple)
    """ 
    Remove all duplicates from a list of geometrical entities. 
    
    Designed to be used with instances of one of the geometrical classes
    (Point, Curve, LineLoop and Surface).
    It should be noted that the equality operator has been overidden for these classes.
    For the LineLoop instances, it takes into account the lineloop direction (clockwise/anticlockwise).

    Note
    --------
    Since the __eq__ method has been overridden in the definition of the geometrical classes,
    their instances are not hashable.
    Faster solution for hashable objects : 
    https://stackoverflow.com/questions/7961363/removing-duplicates-in-lists/7961425#7961425

    """
    unique = list()
    for x in ent_list:
        for y in unique:
            if x == y:
                break
        else:
            unique.append(x)
    return unique

#***
#*** Affine transformations on LineLoops, Curves and Points
#***

#? Test : utilisation d'une factory
# def point_reflection(geo_ent, center):
#     """ Renvoie un nouvel objet géométrique obtenu par symétrie centrale.
    
#     geo_ent : instance of a geometrical class Point, Curve and its subclass or LineLoop
#     """
#     if geo_ent.tag:
#         raise NotImplementedError("For now a geometrical properties of a geometrical entity cannot be modified if this entity has already been added to the gmsh geometry model.") 
#     if isinstance(geo_ent, Point):
#         coord = -(geo_ent.coord - center.coord) + center.coord
#         new_ent = Point(coord)
#     if isinstance(geo_ent, Curve):
#         pts = [point_reflection(pt, center) for pt in geo_ent.def_pts]
#         if isinstance(geo_ent, Line):
#             new_ent = Line(*pts)
#         if isinstance(geo_ent, Arc):
#             new_ent = Line(*pts)
#     if isinstance(geo_ent, LineLoop):
#         if geo_ent.sides:
#             crv = [point_reflection(crv, center) for crv in geo_ent.sides]
#             new_ent = LineLoop(crv, explicit=True)
#         else:
#             pts = [point_reflection(pt, center) for pt in geo_ent.vertices]
#             new_ent = LineLoop(pts, explicit=False)
#         if geo_ent.info_offset:
#             new_ent.info_offset = True
#             new_ent.offset_dpcmt = geo_ent.offset_dpcmt
#     return new_ent

def geo_transformation_factory(pt_coord_fctn):
    """
    A partir d'une fonction élémentaire qui indique comment les coordonnées d'un point sont transformées, cette factory renvoie une fonction capable d'appliquer cette transformation sur une instance de Point, Line, Arc ou LineLoop.

    """
    def transformation(geo_ent, *args, **kwargs):
        """ Renvoie un nouvel objet géométrique obtenu par symétrie centrale.
        geo_ent : instance of a geometrical class Point, Curve and its subclass or LineLoop
        """
        if geo_ent.tag:
            raise NotImplementedError("For now a geometrical properties of a geometrical entity cannot be modified if this entity has already been added to the gmsh geometry model.") 
        if isinstance(geo_ent, Point):
            coord = pt_coord_fctn(geo_ent.coord, *args, **kwargs)
            new_ent = Point(coord)
        if isinstance(geo_ent, Curve):
            pts = [transformation(pt, *args, **kwargs) for pt in geo_ent.def_pts]
            if isinstance(geo_ent, Line):
                new_ent = Line(*pts)
            if isinstance(geo_ent, Arc):
                new_ent = Arc(*pts)
        if isinstance(geo_ent, LineLoop):
            if geo_ent.sides:
                crv = [transformation(crv, *args, **kwargs) for crv in geo_ent.sides]
                new_ent = LineLoop(crv, explicit=True)
            else:
                pts = [transformation(pt, *args, **kwargs) for pt in geo_ent.vertices]
                new_ent = LineLoop(pts, explicit=False)
            if geo_ent.info_offset:
                new_ent.info_offset = True
                new_ent.offset_dpcmt = geo_ent.offset_dpcmt
        return new_ent
    return transformation

def pt_reflection_basis(pt_coord, center):
     return center.coord - (pt_coord - center.coord)
#base_point_reflection = lambda pt_coord, center: -(pt_coord - center.coord) + center.coord
point_reflection = geo_transformation_factory(pt_reflection_basis)

def pln_reflection_basis(pt_coord, pln_pt, pln_normal):
    """
    Symétrie mirroir, en 3D, par rapport à un plan. 
    Le plan est défini par une normale et un point contenu dans ce plan. 
    """
    #?laisser la possibilité d'utiliser directement un np.array ou une liste pour donner les coordonnées pour plus de facilité d'utilisation ?
    if isinstance(pln_pt, Point): 
        pln_pt = pln_pt.coord
    else:
        pln_pt = np.asarray(pln_pt) # Array interpretation of a. No copy is performed if the input is already an ndarray 
    n = unit_vect(pln_normal)
    return (pt_coord - pln_pt) - 2*(pt_coord - pln_pt).dot(n) * n + pln_pt
plane_reflection = geo_transformation_factory(pln_reflection_basis)

def translation_basis(pt_coord, vect):
    """
    vect : 1-D array-like
        The 3 components of the vector that entirely define the translation.
    """
    vect = np.asarray(vect)
    return pt_coord + vect
translation = geo_transformation_factory(translation_basis)