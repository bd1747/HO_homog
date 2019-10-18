# coding: utf8
"""
Created on 09/10/2018
@author: baptiste

Definition of the classes : LineLoop, PlaneSurface and AbstractSurface.

PlaneSurface:
Object designed to represent geometrical entitites of dimension two
and instantiate them in a gmsh model.
"""

from . import factory, np, logger, model
from .tools import round_corner, offset
from .curves import Line, AbstractCurve


def round_corner_2_sides(result_list):
    """ Permet de traiter les résultats d'une opération round_corner
    appliquée en série sur un ensemble de sommets.
    Une polyligne composée de segments et d'arc est composée.
    """
    sides = list()
    # ? à la bonne place ?
    for i, rslt in enumerate(result_list):
        new_line = rslt[0]
        new_line.def_pts[0] = result_list[i - 1][1].def_pts[-1]
        # Correction pour que le segment commence à la fin de l'arc précédent.
        new_arc = rslt[1]
        sides.extend([new_line, new_arc])
    return sides


class LineLoop(object):
    """
    Définit une courbe fermée, composée d'entitées géométriques 1D (Line, Arc...).
    """

    def __init__(self, elements, explicit=False):
        """
        La LineLoop peut être créée à partir :
            - d'une liste de sommets,
            - ou d'une liste d'objets Line/Arcs (explicit = True)
        """

        self.info_offset = False
        # ! A remplacer par quelque chose de mieux, comme l'utilisation de l'attribut "vertices" #noqa
        if explicit:
            self.sides = elements
            self.vertices = list()
        else:
            self.vertices = elements
            self.sides = list()
        self.tag = None

    def __eq__(self, other):
        """
         Opérateur de comparaison == surchargé pour les objets de la classe LineLoop
         Si la LineLoop n'est définie que par ses sommets :
            True ssi les listes de sommets sont égales, à un décalage d'indice près.
         Si la LineLoop est aussi définie par des Line/Arc :
            True ssi l'ensemble des éléments 1D qui composent la LineLoop est
            identique à celui de la LineLoop comparée.
         L'orientation est prise en compte.

         """
        if not isinstance(other, LineLoop):
            return False
        if self.sides or other.sides:
            # Si l'une des deux LineLoops est définie par des Line/Arc, comparaison au niveau de ces éléments. #noqa
            if len(self.sides) != len(other.sides):
                return False
            else:
                for elmt_1D in self.sides:
                    for other_elmt in other.sides:
                        test = elmt_1D == other_elmt
                        if test:
                            break
                    else:
                        # Aucun break déclenché, i.e. si l'un des cote de la lineloop courante n'appartient pas à la LineLoop comparée #noqa
                        return False
                else:
                    return True
        if len(self.vertices) != len(other.vertices):
            return False
        else:
            for shift in range(len(self.vertices)):
                if all(
                    p == other.vertices[i - shift] for i, p in enumerate(self.vertices)
                ):
                    return True
            else:
                return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def plot2D(self, color="black"):
        """Représenter la polyligne dans un plot matplotlib.
        Disponible seulement en 2D pour l'instant."""
        if not self.sides:
            self.vertices_2_sides()
        for elmt in self.sides:
            elmt.plot2D(color)

    def add_gmsh(self):
        if self.tag:
            return None
        if not self.sides:
            self.vertices_2_sides()
        for elmt in self.sides:
            if not elmt.tag:
                elmt.add_gmsh()
        self.tag = factory.addCurveLoop([elmt.tag for elmt in self.sides])
        return self.tag

    def reverse(self):
        self.sides.reverse()
        for elmt in self.sides:
            elmt.def_pts.reverse()
        self.vertices.reverse()

    def offset(self, t, method="vertex"):
        """ Opération d'offset appliquée sur tout les sommets de la LineLoop.

        Cette opération doit donc être faite assez tôt,
        avant que les Line/Arc composant la LineLoop soient créés.
        """
        assert not self.sides
        # Si on est déjà en présence de Lines, il est trop tard pour faire l'offset de cette façon #noqa
        new_vrtces = [None] * (len(self.vertices))
        self.info_offset = True
        for i in range(len(self.vertices)):
            new_vrtces[i - 1] = offset(
                self.vertices[i - 1], self.vertices[i - 2], self.vertices[i], t, method=method
            )
        self.offset_dpcmt = [
            np.linalg.norm(new.coord - prev.coord)
            for new, prev in zip(new_vrtces, self.vertices)
        ]
        # TODO  : regarder où c'est utiliser et si on peut avoir quelque chose de plus clair #noqa
        self.vertices = new_vrtces

    def round_corner_explicit(self, radii):
        """Opération d'arrondi des angles appliquée à tous les sommets du polygone.
        Les rayons sont indiqués de manière explicite, sous forme d'une liste.
        Liste de longueur 1 pour un rayon uniforme.
        """
        # TODOC
        if isinstance(radii, list):
            radii = [radii[i % len(radii)] for i in range(len(self.vertices))]
        else:
            radii = [radii] * (len(self.vertices))
        result_1D = list()
        for i in range(len(self.vertices)):
            result_1D.append(
                round_corner(
                    self.vertices[i - 1],
                    self.vertices[i - 2],
                    self.vertices[i],
                    radii[i - 1],
                    False,
                    False,
                )
            )
        self.sides = round_corner_2_sides(result_1D)

    def round_corner_incircle(self, radii):
        """ Opération d'arrondi des angles appliquée à tous les sommets du polygone.
        La méthode du cercle inscrit est utilisée.
        radii = liste de rayons à utiliser ou valeur (float) si rayon uniforme.
        Une liste de rayons de cercles inscrits peut être indiquée,
        liste de longueur 1 pour un rayon uniforme.
        Si la longueur de la liste de rayon est ni 1 ni égale au nombre de sommets,
        un modulo est utilisé.
        """
        # TODOC
        if isinstance(radii, list):
            effect_R = [radii[i % len(radii)] for i in range(len(self.vertices))]
        else:
            effect_R = [radii] * (len(self.vertices))
        if self.info_offset:
            effect_R = [
                R - offset_d for R, offset_d in zip(effect_R, self.offset_dpcmt)
            ]
            # ! ESSAI
        result_1D = list()
        for i in range(len(self.vertices)):
            result_1D.append(
                round_corner(
                    self.vertices[i - 1],
                    self.vertices[i - 2],
                    self.vertices[i],
                    effect_R[i - 1],
                    True,
                    False,
                )
            )
        self.sides = round_corner_2_sides(result_1D)

    def vertices_2_sides(self):
        """ Méthode permettant de générer les segments reliant les sommets.
        Si une opération round_corner est utilisé, cette opération est inutile."""
        if self.sides:
            logger.warning(
                "Warning : attribut sides d'une LineLoop écrasé "
                "lors de l'utilisation de la méthode vertices_2_sides."
            )
        self.sides = [
            Line(self.vertices[i - 1], self.vertices[i])
            for i in range(len(self.vertices))
        ]


class PlaneSurface(object):
    """
    Calque de la fonction Plane Surface native de gmsh
    Créée à partir d'une LineLoop définissant le contour extérieur
    et, si nécessaire, de line loops définissant des trous internes
    """

    def __init__(self, ext_contour, holes=[]):
        self.ext_contour = ext_contour
        self.holes = holes
        self.tag = None
        self.boundary = ext_contour.sides + [crv for h in holes for crv in h.sides]
        # Pour favoriser le duck typing ?

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
        all_loops = (
            [self.ext_contour] if not self.holes else [self.ext_contour] + self.holes
        )
        for ll in all_loops:
            if not ll.tag:
                ll.add_gmsh()
        self.tag = factory.addPlaneSurface([ll.tag for ll in all_loops])


class AbstractSurface(object):
    """
    Surface dont on ne connait rien à part le tag.
    Une surface existante dans le modèle gmsh peut être identifiée à l'aide de l'API
    puis représentée par une instance de AbstractSurface.
    Il s'agit par exemple du résulat d'une opération booléenne.
    Par contre, ses bords sont a priori inconnus.
    """

    def __init__(self, tag):
        self.tag = tag
        self.boundary = []

    def get_boundary(self, recursive=True):
        """
        Récupérer les tags des entitées géométriques 1D qui composent le bord.

        Parameters
        ----------
        recursive : bool, optional
            If True, the boundaries of the 1-D entities that form the boundary
            of the AbstractSurface instance are also extracted from the gmsh model.
            Instances of Point are created to represent them.

        """
        self.boundary = AbstractSurface.get_surfs_boundary(self, recursive=recursive)

    @staticmethod
    def get_surfs_boundary(surfs, recursive=True):
        """
        Get the tags of all the 1D geometry entities that form the boundary of a surface
        or a group of surfaces.

        Parameters
        ----------
        recursive : bool, optional
            If True, the boundaries of the 1D entities are also extracted
            from the gmsh model.
            Instances of Point are created to represent them.

        """
        def_crv = []
        try:
            for s in surfs:
                if not s.tag:
                    s.add_gmsh()
            dim_tags = [(2, s.tag) for s in surfs]
        except TypeError:
            if isinstance(surfs, (PlaneSurface, AbstractSurface)):
                if not surfs.tag:
                    surfs.add_gmsh()
                dim_tags = (2, surfs.tag)
            else:
                raise (TypeError)
        boundary_ = model.getBoundary(
            dim_tags, combined=True, oriented=False, recursive=False
        )
        for dimtag in boundary_:
            if dimtag[0] != 1:
                logger.warning(
                    "Unexpected type of geometrical entity "
                    f"in the boundary of surfaces {dim_tags}"
                )
                continue
            new_crv = AbstractCurve(dimtag[1])
            if recursive:
                new_crv.get_boundary()
            def_crv.append(new_crv)
        return def_crv


def surface_bool_cut(body, tool):
    """
    Boolean operation of cutting performed on surfaces.

    Remove the aeras taken by the tool entities from the body surface.
    Removing a set of geometrical entities 'tools' at once is possible.
    The removeObject and removeTool parameters of the gmsh API function
    are set to False in order to keep the consistency between
    python geometrical objects and the gmsh geometrical model as far as possible.

    Parameters
    ----------
    body : PlaneSurface or AbstractSurface
        Main operand of the cut operation.
    tool : instance of PlaneSurface/AbstractSurface or list of instances of them
        Several tool areas can be removed to the body surface at once.
        To do this, the tool parameter must be a list.

    Return
    ----------
    cut_surf : AbstractSurface
        Python object that represents the surface obtained with the boolean operation.
        This will be a degenerate instance with only a tag attribut
        and a boundary attribut that can be evaluate later.
    """

    if not body.tag:
        body.add_gmsh()
    if not tool:  # * =True if empty list
        logger.warning(
            "No entity in the tool list for boolean cut operation."
            "The 'body' surface is returned."
        )
        return [body]
    try:
        _ = (element for element in tool)
    except TypeError:
        logger.debug("tool convert to list for boolean operation.")
        tool = [tool]
    for t in tool:
        if not t.tag:
            t.add_gmsh()
    output = factory.cut(
        [(2, body.tag)],
        [(2, t.tag) for t in tool],
        removeObject=False,
        removeTool=False,
    )
    logger.debug(f"Output of boolean operation 'cut' on surfaces : {output}")
    new_surf = list()
    for entity in output[0]:
        if entity[0] == 2:
            new_surf.append(AbstractSurface(entity[1]))
        else:
            logger.warning(
                "Some outputs of a cut boolean operation are not surfaces and"
                "therefore are not returned."
                f"\n Complete output from the API function : {output}"
            )
    return new_surf


def surface_bool_intersect(body, tool):
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
    ops_output = []
    for t in tool:
        outpt = factory.intersect(
            [(2, body.tag)], [(2, t.tag)], removeObject=False, removeTool=False
        )
        if outpt[0]:
            ops_output.append(outpt)
        else:  # Tool entirely outside of body or entirely inside.
            t_copy_dimtag = factory.copy([(2, t.tag)])
            factory.synchronize()  # * Peut être supprimé
            outpt = factory.intersect(
                [(2, body.tag)], t_copy_dimtag, removeObject=False, removeTool=True
            )
            if outpt[0]:
                ops_output.append(outpt)
    new_surf = []
    for outpt in ops_output:
        if outpt[0][0][0] == 2:
            new_surf.append(AbstractSurface(outpt[0][0][1]))
        else:
            warn_msg = (
                "Some entities that result from a intersection boolean operation "
                "are not surfaces and therefore are not returned. \n"
                f"Complete output from the API function : \n {ops_output}"
            )
            logger.warning(warn_msg)
    return new_surf
