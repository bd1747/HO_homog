# coding: utf8
"""
Created on 22/10/2018
@author: baptiste

Définition de classes d'objets géométriques et de fonctions permettant de créer un modèle géométrique de RVE dans gmsh.

Outils pour controler la partie maillage dans gmsh, en complément des outils utiles pour la construction du modèle géoémtrique.
Deux fonctionnalités :
        - Raffinement du maillage autour de points d'intérêts, en utilisant des champs scalaire pour prescire la taille caractéristiques des éléments.
        - Maillage "périodique" : identique sur les parties du bords du RVE en vis à vis.
"""

import logging

import numpy as np

import ho_homog.geometry as geo
import gmsh

# nice shortcuts
model = gmsh.model
api_field = model.mesh.field

logger = logging.getLogger(__name__)


class Field(object):
    """
    Représentation des champs scalaires utilisés pour prescrire
    la taile caractéristique des éléments dans gmsh.

    Tutorial : gmsh-4.0.2-Linux64/share/doc/gmsh/demos/api/t10.py
    """

    def __init__(self, f_type, parent_fields=[]):
        self.tag = None
        self.f_type = f_type
        self.parents = parent_fields

    def set_params(self):
        """Cette méthode sera précisée pour chaque type de champ de scalaire."""
        pass

    def add_gmsh(self):
        """
        Fonction générique pour l'ajout d'un characteristic length field au modèle gmsh.
        """

        if self.tag:
            return None
        if self.parents:
            for p_field in self.parents:
                if not p_field.tag:
                    p_field.add_gmsh()
        self.tag = api_field.add(self.f_type)
        self.set_params()
        return self.tag


class AttractorField(Field):
    """
    Field de type Attractor. Calcul la distance entre un point courant du domaine sur lequel le maillage doit être défini et les attracteurs qui peuvent être des Points, des Lines ou des Arcs.
    Paramètres :
            points : liste d'instances de Point utilisés comme attracteurs;
            curves : liste d'instances de Line ou Arc utilisés comme attracteurs;
            nb_pts_discretization : Nb de points utilisés pour la discrétisation de chaque élément 1D de 'curves'.
    """

    def __init__(
        self, points=[], curves=[], nb_pts_discretization=10
    ):  #! C'est nb_pts_discretization - 2 points pris à l'intérieur de la courbe !
        Field.__init__(self, "Attractor")
        self.points = points if points else None
        self.curves = curves if curves else None
        if curves:
            self.nb_pts_discret = nb_pts_discretization

    def set_params(self):
        if self.points:
            for pt in self.points:
                # ? La méthode add_gmsh contient déjà un test logique pour s'assurer que le Point n'est pas déjà dans le modèle. Est-ce que c'est mieux d'appeler la méthode et de faire le test de la méthode ou de faire un test, puis d'appeler la méthode (qui contient un second test) ?
                if not pt.tag:
                    pt.add_gmsh()
            api_field.setNumbers(self.tag, "NodesList", [pt.tag for pt in self.points])
        if self.curves:
            api_field.setNumber(self.tag, "NNodesByEdge", self.nb_pts_discret)
            for crv in self.curves:
                if not crv.tag:
                    crv.add_gmsh()
            api_field.setNumbers(
                self.tag, "EdgesList", [crv.tag for crv in self.curves]
            )


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

    def __init__(
        self, attract_field, d_min, d_max, lc_min, lc_max, sigmoid_interpol=False
    ):
        Field.__init__(self, "Threshold", [attract_field])
        self.d = (d_min, d_max)
        self.lc = (lc_min, lc_max)
        self.sigmoid = sigmoid_interpol

    def set_params(self):
        api_field.setNumber(self.tag, "IField", self.parents[0].tag)
        api_field.setNumber(self.tag, "LcMin", self.lc[0])
        api_field.setNumber(self.tag, "LcMax", self.lc[1])
        api_field.setNumber(self.tag, "DistMin", self.d[0])
        api_field.setNumber(self.tag, "DistMax", self.d[1])


class RestrictField(Field):
    """
    'Restrict the application of a field to a given list of geometrical points, curves, surfaces or volumes.'

    #? Inclure les boudaries des entités géométriques sélectionnées ?
    """

    def __init__(self, inpt_field, points=[], curves=[], surfaces=[]):
        Field.__init__(self, "Restrict", [inpt_field])
        self.points = points if points else None
        self.curves = curves if curves else None
        self.surfaces = surfaces if surfaces else None

    def set_params(self):
        api_field.setNumber(self.tag, "IField", self.parents[0].tag)
        if self.points:
            for pt in self.points:
                if not pt.tag:
                    pt.add_gmsh()
            api_field.setNumbers(
                self.tag, "VerticesList", [pt.tag for crv in self.points]
            )
        if self.curves:
            for crv in self.curves:
                if not crv.tag:
                    crv.add_gmsh()
            api_field.setNumbers(
                self.tag, "EdgesList", [crv.tag for crv in self.curves]
            )
        if self.surfaces:
            for srf in self.surfaces:
                if not srf.tag:
                    srf.add_gmsh()
            api_field.setNumbers(
                self.tag, "FacesList", [srf.tag for srf in self.surfaces]
            )


class MathEvalField(Field):
    """
     Evaluate a mathematical expression.

    Des champs peuvent être utilisés dans l'expression.
    Dans ce cas, les faire apparaitre dans le paramètre inpt_fields.
    """

    def __init__(self, formula_str, inpt_fields=[]):
        Field.__init__(self, "MathEval", inpt_fields)
        self.formula = formula_str

    def set_params(self):
        api_field.setString(self.tag, "F", self.formula)


class MinField(Field):
    """
    """

    def __init__(self, inpt_fields):
        Field.__init__(self, "Min", inpt_fields)

    def set_params(self):
        api_field.setNumbers(self.tag, "FieldsList", [f.tag for f in self.parents])


# ? Est-ce que emballer tout ça dans un gros objet facilite l'utilisation et/ou la compréhension ? Pas sûr... même plutôt tendance à penser le contraire.
# * Sous forme d'une fonction
def set_mesh_refinement(
    d_min_max,
    lc_min_max,
    attractors={"points": [], "curves": []},
    nb_pts_discretization=10,
    sigmoid_interpol=False,
    restrict_domain={"points": [], "curves": [], "surfaces": []},
):
    """
    Create the fields that are required to impose mesh refinement constraints around some selected points or curves.
    Return the major field which that should be used in subsequent operations on fields.
    The application of the refinement constraint can be restricted to a part of the geometrical model.

    Parameters
    ----------
    d_min_max : list of two floats
        Distances from the attractors that delimit the area where the element size is interpolated between lc_min and lc_max.
    lc_min_max : list of two floats
        The first value, lc_min, is the caracteristic element size inside balls of raduis d_min and centered at each attractor point;
        The second value, lc_max, is the caracteristic element size outside balls of raduis d_max and centered at each attractor point.
            prescribed element size = lc_min if distance from the nearest attractor <= d_min
                                    = lc_max if distance from the nearest attractor >= d_max
                                    = interpolation between lc_min and lc_max else (if d_min < distance < d_max)
    attractors : dictionnary, two possible keys "points" and "curves"
        The geometrical entities around which the mesh must be finer, called attractors.
        Points and curves can be used as attractors.
        Points have to be instances of the Point class and curve have to be instances of the Line or Arc classes.
    nb_pts_discretization : float, optional
        If curves are used as attractors, each curve is replaced by a discrete set of equidistant points that are on the curve.
        The distance from those points is used to compute the distance from the attractor curves during the mesh generation.
        nb_pts_discretization is the number of those points.
    sigmoid_interpol = bool, optional
        If False, the element size that is prescribe between d_min and d_max is calculated with a linear interpolation between lc_min and lc_max.
        If True, a sigmoid function is used for this interpolation.
    restrict_domain : dictionnary, optional
        It is possible to impose The application of the refinement constraint can be restricted to selected geometrical entities.
        Geometrical points, curves and surfaces can be given, in lists, using the three keys : 'points', 'curves' and 'surfaces'. They have to be instances of the Point, Line/Arc and PlaneSurface classes respectively.

    Returns
    -------
    major_field : fied
        Instance of a subclass of Field that entirely characterize the refinement constraint.
        If the application of the refinement constraint is restricted to geometrical entities then this field is an instance of the RestrictField class, else it is an instance of ThresholdField.

    """
    try:
        inpt_points = attractors["points"]
    except KeyError:
        inpt_points = []
    try:
        inpt_curves = attractors["curves"]
    except KeyError:
        inpt_curves = []
    attract_field = AttractorField(inpt_points, inpt_curves, nb_pts_discretization)
    d_min, d_max = d_min_max
    lc_min, lc_max = lc_min_max
    threshold_field = ThresholdField(attract_field, d_min, d_max, lc_min, lc_max)
    if restrict_domain:
        try:
            rstrc_points = restrict_domain["points"]
        except KeyError:
            rstrc_points = []
        try:
            rstrc_curves = restrict_domain["curves"]
        except KeyError:
            rstrc_curves = []
        try:
            rstrc_surf = restrict_domain["surfaces"]
        except KeyError:
            rstrc_surf = []
        restrict = True if (rstrc_points or rstrc_curves or rstrc_surf) else False
    else:
        restrict = False
    if restrict:
        restrict_field = RestrictField(
            threshold_field, rstrc_points, rstrc_curves, rstrc_surf
        )
        return restrict_field
    else:
        return threshold_field


def set_background_mesh(fields):
    """
    Set the background scalar field that will be used in order to prescribe all the element size constraints during the mesh generation process.

    Only one background field can be given for the mesh generation. (gmsh reference manual 26/09/2018, 6.3.1 Specifying mesh element size)
    If multiple fields are specified in the unique parameter, the related element size constrains will be combined and reduce to a single field by means of a Minimum operation.
    The whole domain on which a mesh is going to be generated should be covered by at least one specified element size field.

    Parameters
    ----------
    fields : a single field object or a list of fields
        The specified fields must be instances of the subclasses of the Field base class.
        It (or they) describe the element size that it is desired over the whole material domain on which a mesh is going to be generated.

    """
    if not isinstance(fields, list):
        final_field = fields
    else:
        final_field = MinField(fields)
    final_field.add_gmsh()
    api_field.setAsBackgroundMesh(final_field.tag)
    return final_field


# TODO : Pourquoi pas mettre dans un fichier à part
# *#*#*
# * Tools for creating a periodic 2D mesh (may be generalized to 3D later)
# *#*#*


def translation2matrix(v, dist=None):
    """
    Return the affine transformation matrix that represent a given translation.

    If only the value of the parameter 'v' is given, then the represented translation is the translation by the vector 'v'. Else, if both 'v' and 'dist' parameters are specified, it is the translation in the direction of 'v' (oriented) by the distance 'dist' (algebric).

    Parameters
    ----------
    v : 1-D numpy array
        Vector that entirely define the translation or only the direction if the second parameter is specified.
        The user can implicitely define the translation in the (e_x, e_y) plane and give a (2,) array for the v value.
    d : float, optional
        Distance of translation.

    Returns
    -------
    transform_matx : list of 16 float values
        The affine transformation matrix A of shape 4x4 that represents the translation.
        For a given affine transformation T mapping R^3 to R^3, represented by the tranformation matrix A :
                x' = T(x)
            <=> (x', y', z', 1)^T = A cdot (x, y, z, 1)^T
        The 4x4 matrix is flattened in row-major order and returned as a list.

    See Also
    --------
    https://en.wikipedia.org/wiki/Transformation_matrix for more information about the matrix representation of affine transformations.

    Examples
    --------
    >>> msh.translation2matrix(np.array([5, 0]))
    [1.0, 0.0, 0.0, 5.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    >>> msh.translation2matrix(np.array([1, 2, 3]))
    [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0]
    >>> mtx3 = msh.translation2matrix(np.array([1, 1, 0]), 1)
    >>> msh.translation2matrix(np.array([1, 1, 0]), 1)
    [1.0, 0.0, 0.0, 0.7071067811865475, 0.0, 1.0, 0.0, 0.7071067811865475, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    """
    if dist:
        v_ = geo.unit_vect(v)
        v_ = dist * v_
    else:
        v_ = v
    if v_.shape == (3,):
        v_ = np.append(v_, [0])
    elif v_.shape == (2,):
        v_ = np.append(v_, [0, 0])
    else:
        raise TypeError("The numpy array can not correspond to a translation vector.")
    transform_matx = np.identity(4)
    transform_matx[:, 3] += v_
    return transform_matx.flatten().tolist()


def set_periodicity_pairs(slaves, masters, translation_v=np.array(())):
    """
    A rédiger #TODO

    Il n'est pas nécessaire de donner le vecteur translation explicitement,
    on se sert des points des éléments de slaves et master pour le définir
    """
    if (not slaves) or (not masters):
        logger.warning(
            "No slave or master geometrical entities are given for the definition of a periodicity constraint."
        )
        logger.warning("This periodicity constraint will be ignored.")
        return False

    all_entities = slaves + masters
    if all(isinstance(e, geo.Curve) for e in all_entities):
        geo_dim = 1
    elif all(
        isinstance(e, (geo.AbstractSurface, geo.PlaneSurface)) for e in all_entities
    ):
        geo_dim = 2
    else:
        raise TypeError(
            "For set_periodicity_pairs, all entities must be of the same dimension. "
            "1D and 2D geometrical entities supported."
        )
    for e in all_entities:
        if not e.tag:
            e.add_gmsh()

    if translation_v.any():
        vect = translation_v
    elif geo_dim == 1:
        vect = slaves[0].def_pts[0].coord - masters[0].def_pts[0].coord
    else:
        raise ValueError(
            "For 2D entities the translation vector must be explicitely given."
        )

    model.mesh.setPeriodic(
        geo_dim,
        [s.tag for s in slaves],
        [m.tag for m in masters],
        translation2matrix(vect),
    )
    return True


def sort_function_factory(dir_v):
    """
    Info : https://en.wikipedia.org/wiki/Closure_(computer_programming)
    """

    def sort_function(curve):
        return np.dot(curve.def_pts[0].coord + curve.def_pts[-1].coord, dir_v)

    return sort_function


# ! A changer d'endroit : mettre dans geometry
def order_curves(curves, dir_v, orientation=False):
    """
    Ordonne une liste de courbes.
    dir_v correspond globalement à la direction des courbes. Le produit scalaire avec ce vecteur directeur est utilisé pour définir l'ordre des courbes.
    orientation:
        Si True, l'orientation des courbes est corrigée de sorte à ce qu'elles soient toutes dans le même sens.

    Return
    ------
    None
        Liste de curves modifiée sur place.

    Info
    "La seul différence est que sort() retourne None et modifie sur place, tandis que sorted() retourne une nouvelle liste. sorted() est un peu moins performant."
    http://sametmax.com/ordonner-en-python/

    """

    curves.sort(key=sort_function_factory(dir_v))
    if orientation:
        for c in curves:
            if np.dot(c.def_pts[0].coord, dir_v) > np.dot(c.def_pts[-1].coord, dir_v):
                c.def_pts.reverse()
                logger.debug(
                    "Orientation d'une courbe inversée."
                    f"Nouvel ordre des definition points : {[p.coord for p in c.def_pts]}"
                )
    return None

