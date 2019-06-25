# coding: utf8
"""
Created on 09/10/2018
@author: baptiste

Definition of the classes : Curve, Line, Arc and AbstractCurve.
Objects designed to represent geometrical entitites of dimension one
and instantiate them in a gmsh model.
"""

import logging
import warnings
from logging.handlers import RotatingFileHandler
from pathlib import Path

from . import factory, logger, model, np, plt
from .point import Point

bndry_logger = logging.getLogger("bndry")
bndry_logger.setLevel(logging.DEBUG)
bndry_logger.propagate = False
log_path = Path("~/ho_homog_log/gmsh_getBoundary_output.log").expanduser()
if not log_path.parent.exists():
    log_path.parent.mkdir(mode=0o777, parents=True)
file_handler = RotatingFileHandler(str(log_path), "a", 1000000, 1)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s")
)
bndry_logger.addHandler(file_handler)
bndry_logger.warning("***Output of the getBoundary function of the gmsh API***")
bndry_logger.warning(
    """
*****
This logger and the associated handler are a hacky workaround to avoid errors
related to gmsh.model.occ.getBoundary(). Therefore they should not be removed
or deactivated.
*****"""
)


class Curve(object):
    """Superclass that is used to define the other classes : Line, Arc and AbstractCurve.

    It is designed to represent geometrical entities of dimension one.
    """

    def __init__(self, def_pts_list, gmsh_api_add_function):
        self.def_pts = def_pts_list
        self.tag = None
        self.gmsh_constructor = gmsh_api_add_function
        Curve.all_instances.append(self)

    def __eq__(self, other):
        """
        Return True if and only if :
        - both self and other are instances of the same subclass,
        AND
        - The coordinates of the points that are used to define
        these two Lines (or Arcs) are equal.

        """
        if not type(other) is type(self):
            return False
        return all(p == q for p, q in zip(self.def_pts, other.def_pts))

    def __ne__(self, other):
        return not self.__eq__(other)

    def add_gmsh(self):
        """Instantiate a geometrical entity in the current gmsh model"""

        if self.tag:
            return None
        for pt in self.def_pts:
            if not pt.tag:
                pt.add_gmsh()
        self.tag = self.gmsh_constructor(*[p.tag for p in self.def_pts])
        return self.tag

    # IDEA
    # ? @property
    # ? def tag(self):
    # ?     if self.__tag is None:
    # ?         self.add_gmsh()
    # ?     return self.__tag

    def reverse(self):
        """ Peut-être utile au moment de définir les contraites de périodicité.
        # TODO À tester !!!
        """
        self.def_pts.reverse()
        if self.tag:
            self.tag *= -1


class Line(Curve):
    """A line is defined by 2 instances of Point (start + end)"""

    def __init__(self, start_pt, end_pt):
        Curve.__init__(self, [start_pt, end_pt], factory.addLine)

    def __str__(self):
        """Affichage plus clair des coordonnées des points de départ et d'arrivée."""
        prt_str = (
            f"Line {self.tag if self.tag else '--'}, "
            f"Point tags : start {self.def_pts[0].tag}, end {self.def_pts[1].tag}"
        )
        return prt_str

    def length(self):
        return np.linalg.norm(self.def_pts[0].coord - self.def_pts[1].coord)

    def direction(self):
        """ Renvoie un vecteur unitaire correspondant à la direction de la ligne"""
        return 1 / self.length() * (self.def_pts[1].coord - self.def_pts[0].coord)

    def plot2D(self, color="black"):
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
        """ Crée un arc de cercle.

        Vérification préalable de l'égalité des distances centre<->extrémités.
        """

        d1 = np.linalg.norm(start_pt.coord - center_pt.coord)
        d2 = np.linalg.norm(end_pt.coord - center_pt.coord)
        np.testing.assert_almost_equal(d1, d2, decimal=10)
        Curve.__init__(self, [start_pt, center_pt, end_pt], factory.addCircleArc)
        self.radius = (d1 + d2) / 2

    def __str__(self):
        prt_str = (
            f"Circle arc {self.tag if self.tag else '--'}, "
            f"Point tags : start {self.def_pts[0].tag}, center {self.def_pts[1].tag}, "
            f"end {self.def_pts[2].tag}"
        )
        return prt_str

    def plot2D(
        self,
        circle_color="Green",
        end_pts_color="Blue",
        center_color="Orange",
        pt_size=5,
    ):
        """Représenter l'arc de cercle dans un plot matplotlib.
        Disponible seulement en 2D pour l'instant."""

        self.def_pts[0].plot(end_pts_color, pt_size)
        self.def_pts[2].plot(end_pts_color, pt_size)
        self.def_pts[1].plot(center_color, pt_size)
        circle = plt.Circle(
            (self.def_pts[1].coord[0], self.def_pts[1].coord[1]),
            self.radius,
            color=circle_color,
            fill=False,
        )
        ax = plt.gca()
        ax.add_patch(circle)


class AbstractCurve(Curve):
    # TODOC

    @staticmethod
    def empty_constructor(*args):
        warnings.warn(
            "Adding an AbstractCurve instance to the gmsh model is not supported. "
            "\n These python objects actually represent unknown geometrical entities"
            " that already exist in the gmsh model.",
            UserWarning,
        )

    def __init__(self, tag):
        """
        Créer une représentation d'une courbe existant dans le modèle Gmsh.

        # ! A corriger
        # ! Lors de l'instantiation, les points extrémités peuvent être donnés,
        # ! soit explicitement, soit par leurs tags.
        """
        Curve.__init__(self, [], AbstractCurve.empty_constructor)
        self.tag = tag

    def get_boundary(self, get_coords=True):
        """
        Récupérer les points correspondants aux extrémités de la courbe
        dans le modèle Gmsh.

        Parameters
        ----------
        coords : bool, optional
            If true, les coordonnées des points extrémités sont aussi récupérés.

        """
        bndry_logger.debug(f"Abstract Curve -> get_boundary. self.tag : {self.tag}")
        def_pts = []
        boundary = model.getBoundary((1, self.tag), False, False, False)
        bndry_logger.debug(
            f"Abstract Curve -> get_boundary. raw API return : {boundary}"
        )
        for pt_dimtag in boundary:
            if not pt_dimtag[0] == 0:
                raise TypeError(
                    f"The boundary of the geometrical entity {self.tag} are not points."
                )
            coords = model.getValue(0, pt_dimtag[1], []) if get_coords else []
            logger.debug(repr(coords))
            new_pt = Point(np.array(coords))
            new_pt.tag = pt_dimtag[1]
            def_pts.append(new_pt)
        self.def_pts = def_pts

    def plot2D(self, color="black"):
        """En 2D seulement. Tracé des points de def_pts, reliés en pointillés"""
        x = [pt.coord[0] for pt in self.def_pts]
        y = [pt.coord[1] for pt in self.def_pts]
        plt.plot(x, y, color=color, linestyle="dashed")
