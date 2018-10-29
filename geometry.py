# -*- coding: utf-8 -*-
"""
Created on 09/10/2018
@author: baptiste

Définition de classes d'objets géométriques et de fonctions permettant de créer un modèle géométrique de RVE dans gmsh.

sources : 
    - https://deptinfo-ensip.univ-poitiers.fr/ENS/doku/doku.php/stu:python:pypoo

"""

import gmsh

# nice shortcuts
model = gmsh.model
factory = model.occ

#TODO : regarder ou mettre ces commandes
# gmsh.initialize() #? A mettre dans un fichier init ????
# #? Uitliser cet argument ? gmsh.initialize(sys.argv) cf script boolean.py
# gmsh.option.setNumber("General.Terminal", 1) #0 or 1 : print information on the terminal

# TODO : à trier
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import math
import copy
import os
import operator

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
    PhysicalEntity.count = 1
    PhysicalEntity.tagDejaPris = set()
    

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
def centroSym(pt, centre):
    """ Renvoie un nouveau point obtenu par symétrie centrale."""
    new_coord = -(pt.coord - centre.coord) + centre.coord
    return Point(new_coord)

def mirrorSym(pt, centre, axe):
    """
    Renvoie un nouveau point obtenu par symétrie mirroir. axe de symétrie décrit par un point et un vecteur
    SEULEMENT EN 2D POUR LE MOMENT
    To do : généraliser 2D et 3D
    """
    new_coord = (2 * (pt.coord - centre.coord).dot(axe) * axe - (pt.coord - centre.coord) + centre.coord)
    return Point(new_coord)

def translat(pt, vect):
    """ Translation d'un point, défini par un vecteur de type np.array
    """
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


class Line(object):
    """Classe définissant une ligne simple caractérisée par :
    - son point de départ
    - son point d'arrivée
    """
    #TODO : Créer une méthode pour inverser le sens de la ligne ?

    def __init__(self, start_pt, end_pt):
        self.def_pts = [start_pt, end_pt]  #? Préférer cette écriture sous forme de List qui peut se généraliser aux arc, par exemple ?
        self.tag = None

    # def __eq__(self, other):
    #     """ Renvoie True ssi les coordonnées des points de départ et d'arrivée sont égaux deux à deux. """

    #     if not isinstance(other, Line):  # Lors de la comparaison d'éléments d'une LineLoop, il se peut qu'une Line soit comparé à un Arc.
    #         return False
    #     return all(p==q for p, q in zip(self.def_pts, other.def_pts))

    def __eq__(self, other):
        """ Renvoie True ssi les coordonnées des points de départ et d'arrivée sont égaux deux à deux.
        Méthode sert aussi pour les objets Arc"""

        if not type(other) is type(
                self):  # Lors de la comparaison d'éléments d'une LineLoop, il se peut qu'une Line soit comparé à un Arc.
            return False
        return all(p == q for p, q in zip(self.def_pts, other.def_pts))

    def __ne__(self, other):
        return not self.__eq__(other)

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
        self.tag = factory.addLine(self.def_pts[0].tag, self.def_pts[1].tag)


class Arc(Line):
    """Classe définissant un arc de cercle caractérisé par :
    - son point de départ
    - son point d'arrivée
    - son centre
    - son rayon
    - son nom
    """

    def __init__(self, start_pt, center_pt, end_pt):
        """ Crée un arc de cercle, après avoir comparé les distances entre le centre et les deux extrémités indiquées.
         https://docs.scipy.org/doc/numpy/reference/generated/numpy.testing.assert_array_almost_equal.html
        """
        #if isinstance(start_pt, Point) and isinstance(end_pt, Point) and isinstance(center_pt, Point):
        #        else:
        #    raise TypeError("Les input ne sont pas au bon format (points)")
        d1 = np.linalg.norm(start_pt.coord - center_pt.coord)
        d2 = np.linalg.norm(end_pt.coord - center_pt.coord)
        np.testing.assert_almost_equal(d1, d2, decimal=10)
        self.radius = (d1 + d2) / 2
        self.def_pts = [start_pt, center_pt, end_pt]
        self.tag = None

    def __str__(self):
        # prt_str = "Arc %s \n Pt début : %s Pt fin : %s Centre : %s" %(self.name, str(self.deb), str(self.fin), str(self.centre))
        prt_str = "Circle arc " + self.tag if self.tag else "--" + ", "
        prt_str += "start point tag : %i , center tag : %i , end point tag : %i" %(self.def_pts[0].tag, self.def_pts[1].tag, self.def_pts[2].tag)
        return prt_str

    def plot(self,  circle_color="Green", end_pts_color="Blue", center_color="Orange", pt_size=5):
        """Représenter l'arc de cercle dans un plot matplotlib. 
        Disponible seulement en 2D pour l'instant."""

        self.def_pts[0].plot(end_pts_color, pt_size)
        self.def_pts[2].plot(end_pts_color, pt_size)
        self.def_pts[1].plot(center_color, pt_size)
        circle = plt.Circle((self.def_pts[1].coord[0],self.def_pts[1].coord[1]), self.radius, color=circle_color, fill=False)
        ax = plt.gca()
        ax.add_patch(circle)

    def add_gmsh(self):
        """Méthode permettant d'ajouter une ligne de script dans un fichier .geo pour l'arc de cercle,
        et si nécesaire les deux points extrémités et le centre.
        """
        if self.tag: # Geometrical entity has already been instantiated in the gmsh model.
            return self.tag # For information purposes only.

        for pt in self.def_pts:
            if not pt.tag:
                pt.add_gmsh()
        self.tag = factory.addCircleArc(self.def_pts[0].tag, self.def_pts[1].tag, self.def_pts[2].tag)


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
    if type(inp_ll) is not list:
        new_pts = [mirrorSym(pt, center, axe) for pt in inp_ll.vertices]
        new_pts.reverse()
        return LineLoop(new_pts, explicit=False)
    else:
        new_ll_list = [mirror_sym_ll(each_ll, center, axe) for each_ll in inp_ll]
        return new_ll_list


def translat_ll(inp_ll, vect):
    """ Doc_string à compléter. Voir mirror_sym_ll pour un fonctionnement similaire. """
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


#TODO : Choisir !!!! ET SURTOUT REGARDER CE QUI EST LE PLUS COH2RENT ! 
#! Choix de passer l'opération booléenne dans une fonction, au lieu d'être une méthode de la classe Surface
#? Est-ce que l'opération booléenne pourrait fonctionner sur d'autres objets que des surfaces ?
# def bool_cut(body, tools, tag=-1):
# """
# Remove the tool entities from the body entity. Removing a set of geometrical entities 'tools' is possible.
# Si plusieurs tools : l'input doit être une liste
# option pour supprimer ou non les tools. 
# """
#     pass
#? SI ON A UNE FONCTION A PART ENTIERE, CEST PEUT ETRE PAS PLUS MAL : LA NOUVELLE SURFACE EST DEFINI SEULEMENT PAR L'OPERATION BOOLEENNE, ON NE CONNAIT PAS SES SOMMETS NI SES COTES, DONC ON CREE UN NOUVEL OBJET PYTHON DIFFERENT DE BODY ????

    def bool_cut(self, tools, remove_tools=False):
    #def boolRemoveGmsh(self, planeSurfToolList, script, delTool=False):
        #TODO : refaire docstring
        """
        Remove the tool entities from the body entity. Removing a set of geometrical entities 'tools' is possible.
        Si plusieurs tools : l'input doit être une liste.
        Option pour supprimer ou non les tools. 

        Opération booléenne de soustraction, sur des objets gmsh de type Plane Surface
        NECESSITE noyau géométrique OpenCASCADE "SetFactory("OpenCASCADE");"
        Possibilité de garder ou de supprimer l'outil (par défaut)
        Rmq : dans gmsh le résultat de l'opération garde le même ID que l'objet autant que possible avec l'option Geometry.OCCBooleanPreserveNumbering
        Cela n'est possible que si l'object est supprimé lors de l'opération.
        """
        #   TO DO: NECESSITE noyau géométrique OpenCASCADE "SetFactory("OpenCASCADE");"
        if not self.in_model:
            self.add_gmsh()
        if isinstance(tools, PlaneSurface):
            tools = [tools]
        assert isinstance(tools, list)
        for t in tools:
            if not t.in_model:
                t.add_gmsh()
            output = factory.cut([(2,self.tag)], [(2,t.tag)], removeObject=True, removeTool=remove_tools) #TODO: si on généralise, c'est là qu'on peut avoir geo_dim
            #? passer directement une liste de tools en input de la fonction ?
            print("output boolean operation cut : ", output) #! Temporaire. Voir ce que renvoie une opération booléenne. Qu'est ce qu'est le 2d output ? 
            new_tag = output[0][0][1]
            if new_tag != self.tag :
                print("[Info] The boolean cut operation change the tag of the surface : surface %i becomes %i" %(self.tag, new_tag))
                self.tag = new_tag
            if remove_tools :
                t.in_model = False
                #! Faire quelquechose sur le tag ???
    #! A TESTER 
    #!  #! #!

    def bool_intersect(self, tools, remove_tools=False):
    #def boolIntersectGmsh(self, planeSurfToolList, script, delTool=False):
        """
        Opération booléenne d'intersection, sur des objets gmsh de type Plane Surface.
        Résultat seulement visible dans gmsh, les objets dans Python ne sont pas modifiés
        Possibilité de garder ou de supprimer l'outil (par défaut)
        """
        if not self.in_model:
            self.add_gmsh()
        if isinstance(tools, PlaneSurface):
            tools = [tools]
        assert isinstance(tools, list)
        for t in tools:
            if not t.in_model:
                t.add_gmsh()
            output = factory.intersect([(2,self.tag)], [(2,t.tag)], removeObject=True, removeTool=remove_tools)
            print("output boolean operation cut : ", output) #! Temporaire
            new_tag = output[0][0][1]
            if new_tag != self.tag :
                print("[Info] The boolean cut operation change the tag of the surface : surface %i becomes %i" %(self.tag, new_tag))
                self.tag = new_tag
            if remove_tools :
                t.in_model = False
                #! Faire quelquechose sur le tag ???
    #! A TESTER 
    #!  #! #!

#TODO : à mieux définir
    # def delGmsh(self, script, recursive=False):
    #     delGmsh(self, "Surface", script, recursive)


class PhysicalEntity(object):
    """
    Physical entity d'un certain type, à préciser en entrée.
    Est identifié par un tag et/ou un ID uniques
    """
    count = 1
    tagDejaPris = set()

    def __init__(self, inpList, inpType, tag=None):
        """
        Inptype : type d'entités géométriques regroupées. Point, Line, Surface ou Volume
        """
        self.idx = copy.deepcopy(PhysicalEntity.count)
        PhysicalEntity.count += 1

        if tag == None:
            self.tag = None
        elif not tag in PhysicalEntity.tagDejaPris:
            self.tag = copy.deepcopy(tag)
            PhysicalEntity.tagDejaPris.add(self.tag)
        else:
            raise ValueError("tag déjà utilisé")

        self.elmts = inpList  # Utiliser un set ou une liste ?
        self.in_script = False
        self.elmtType = inpType

    def addgmsh(self, script):
        if not self.in_script:
            for elmt in self.elmts:
                if not elmt.in_script:
                    elmt.addgmsh(
                        script)  # TOUJOURS LE MEME PROBLEME DE LONGUEUR CARACTERISTIQUE SI CEST UN ENSEMBLE DE POINTS

            outputline = "Physical " + self.elmtType + "("

            if self.tag != None:
                outputline += '"' + self.tag + '", '
            outputline += str(self.idx) + ") = {" + ", ".join([str(elmt.idx) for elmt in self.elmts]) + "};\n"
            script.write(outputline)
            self.in_script = True
        # Modèles :

    #        Physical Line("test") = {3, 2, 4};
    #        Physical Line(88) = {2, 3};
    #        Physical Point("testPote") = {2, 1, 4};
    #        Physical Surface("testSur") = {1};
    #        Physical Surface("test", 1) += {1};

    def add(self, script, elmt):
        """
        ajouter un élément à une physical entity
        """
        if not elmt in self.elmts:
            self.elmts.append(elmt)

            outputline = "Physical " + self.elmtType + "("
            if self.tag != None:
                outputline += '"' + self.tag + '", '
            outputline += str(self.idx) + ") += {" + str(elmt.idx) + "};\n"
            # modèle : Physical Surface("test", 1) += {1};

        else:
            print
            "Element déjà présent dans l'entité physique ", self.tag, str(self.idx)
            print
            elmt

    def setMeshColor(self, script, color):
        """
        Choisir la couleur du maillage qui coincidera avec les éléments géométriques contenus de l'entité physique.
        color : string, nom ( Blue, Black, Red, Orange,...) ou expression RGB "{255,255,0}"
        """
        output = "Color %s { %s {%s};} \n" % (color, self.elmtType, ", ".join([str(elmt.idx) for elmt in self.elmts]))
        script.write(output)
        # Modèles : Color Red { Point{3}; Surface{1}; }; Color {255,255,0} { Line{4}; }; Color White { Line {2, 1, 4}; }

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

def remove_duplicates_ll(geo_entities):
    """
    Supprimer les doublons dans une liste, en conservant l'ordre.
    Fonction conçue pour fonctionner avec des listes de LineLoop.
    """
    filtered = list()
    for nxt_entity in geo_entities:
        for already_entity in filtered:
            if nxt_entity == already_entity:
                break
        else:
            newList.append(elmt)
    #? Faire avec : [nv.append(item) for item in liste if not item in nv] ? comment le test d'appartenance fonctionne avec des classes perso ? A TESTER
    return newList
    #! Fonction à tester !

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

    pt_racc_amt = translat(inp_pt, dist_racc * v_amt)
    pt_racc_avl = translat(inp_pt, dist_racc * v_avl)
    # pt_racc_amt.name = "pt_racc_amt"
    # pt_racc_avl.name = "pt_racc_avl"
    pt_ctr = translat(inp_pt, dist_center * v_biss)

    round_arc = Arc(pt_racc_amt, pt_ctr, pt_racc_avl)
    racc_amt = Line(pt_amt, pt_racc_amt)#, "Ligne de raccord amont")
    racc_avl = Line(pt_racc_avl, pt_avl)#, "Ligne de raccord aval")
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


if __name__ == '__main__':
    test = []

    ### test de la méthode boolRemovegmsh de PlaneSurface ####
    if "planeSurface" in test:
        a, b = 1., 2.

        Incl = [(a, 0.), (0., b), (-a, 0.), (0., -b)]
        Incl = map(np.array, Incl)
        Incl = map(Point, Incl)

        Rect = [(4 * a, 4 * b), (-4 * a, 4 * b), (-4 * a, -4 * b), (4 * a, -4 * b)]
        Rect = map(np.array, Rect)
        Rect = map(Point, Rect)

        surfRect = PlaneSurface(LineLoop(Rect))
        surfIncl = PlaneSurface(LineLoop(Incl))

        scriptTest = "testBoolRemoveSurface.geo"
        with open(scriptTest, 'w') as fileOut:
            geometry_kernel(fileOut, 1)
            surfRect.addgmsh(fileOut)
            surfRect.boolRemoveGmsh([surfIncl], fileOut, delTool=True)

        strCmd = "gmsh " + scriptTest
        os.system(strCmd)  # Ok ! Les deux géométries sont créées et affichées correctement

    #### Test fonction bissectrice offset et angle ####
    vectTests = [[np.array((0., 2.)), np.array((-3. * math.sqrt(2.) / 2., -3. * math.sqrt(2.) / 2.))],
                 [np.array((-1., 0.)), np.array((2. * math.sqrt(3) / 2., 2. * 1. / 2.))],
                 [np.array((-5., -5.)), np.array((5., 5.))], [np.array((3., -2.)), np.array((-3., 2.))]]
    if "bissectrice" in test:
        for vectCouple in vectTests:
            print
            angle_between(vectCouple[0], vectCouple[1], True)
        # Pour bissectrice
        for vectCouple in vectTests:
            P1 = Point(vectCouple[0])
            P2 = Point(vectCouple[1])
            P3 = Point(bissect(vectCouple[0], vectCouple[1]))
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            P1.plot("red")
            Line(Point(), P1).plot("red")
            P2.plot("blue")
            Line(Point(), P2).plot("blue")
            P3.plot("orange")
            Line(Point(), P3).plot("orange")
            plt.axis('equal')
            plt.show()

        # Pour bissectrice 2
        for vectCouple in vectTests:
            P1 = Point(vectCouple[0])
            P2 = Point(vectCouple[1])
            P3 = Point(bissect2(vectCouple[0], vectCouple[1]))
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            P1.plot("red")
            Line(Point(), P1).plot("red")
            P2.plot("blue")
            Line(Point(), P2).plot("blue")
            P3.plot("orange")
            Line(Point(), P3).plot("orange")
            plt.axis('equal')
            plt.show()
        # OK !
    if "offset" in test:
        # Pour offset
        from matplotlib.backends.backend_pdf import PdfPages

        pp = PdfPages('offsetTests.pdf')

        for vectCouple in vectTests:
            P1 = Point(vectCouple[0])
            P2 = Point(vectCouple[1])
            P3 = Point(np.array((0., 0.)))
            P4 = offset2(P3, P1, P2, 0.5)
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            P1.plot("red")
            Line(Point(), P1).plot("red")
            P2.plot("blue")
            Line(Point(), P2).plot("blue")
            P3.plot("orange")
            P4.plot("purple")
            Line(translat(P1, P4.coord - P3.coord), P4).plot("purple")
            Line(translat(P2, P4.coord - P3.coord), P4).plot("purple")
            plt.axis('equal')
            plt.savefig(pp, format='pdf')
            plt.show()
        pp.close()

    #### Test fonction PeriodicLinePaires ####
    if "periodic" in test:
        a, b = 1., 1.
        rect = [(a, b), (-a, b), (-a, -b), (a, -b)]
        rect = [Point(np.array(coord)) for coord in rect]
        contourRect = LineLoop(rect, polygone=True)
        surfRect = PlaneSurface(contourRect)

        script = "Ex_periodic_mesh.geo"
        with open(script, 'w') as fileOut:
            rect[0].addgmsh(fileOut, 0.01)
            surfRect.addgmsh(fileOut)
            perLinePaires(fileOut, [3, 4], [-1, -2], ID=True)
            # OK ! Cela donne bien le résultat attendu, visuellement. Vérifier si les coordonées les noeuds sont corrects ?

            group1 = PhysicalEntity([contourRect.sides[0], contourRect.sides[2]], "Line", tag="groupTest")
            group1.addgmsh(fileOut)
            group1.setMeshColor(fileOut, "{0,255,255}")  # OK

            fileOut.write("Mesh 2; ")  # ATTENTION, genere un maillage mais n'écrit pas de fichier .msh

        strCmd = "gmsh " + script
        os.system(strCmd)

    #### test fonction importLineListFromFile ####
    if "importLine" in test:
        fileTest = "exp_gmsh_bord_haut"
        resultLines = importLineList(fileTest)
        for elmt in resultLines:
            elmt.plot()
            elmt.deb.plot()
            elmt.fin.plot()

