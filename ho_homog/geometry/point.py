# coding: utf8
"""
Created on 09/10/2018
@author: baptiste

Definition of the Point class.
Base geometrical object for model definition in gmsh.
"""

from . import factory, plt, np


class Point(object):
    # TODOC

    def __init__(self, coord=np.array((0.0, 0.0)), mesh_size: float = 0.0):
        """Create a new point.

        Parameters
        ----------
        coord : 1D ndarray (vector), optional
            coordinates of the point, in 2D or 3D.
            By default : np.array((0., 0.))
        mesh_size : float, optional
            Characteristic length imposed for the mesh generation at this point.
            By default : 0
        """
        coord = np.asarray(coord)  # * existing arrays are not copied
        dim = coord.shape[0]
        self.coord = coord
        if dim == 2:
            self.coord = np.append(self.coord, [0.0])
        # ? Choix : on utilise toujours des coordonnés en 3D.
        self.tag = None
        self.mesh_size = mesh_size

    def __repr__(self):
        """Represent a Point object with the coordinates of this point."""
        return f"Pt {self.tag} ({str(self.coord)}) "

    def __eq__(self, other):
        """Opérateur de comparaison == redéfini pour les objets de la classe Point.

        Renvoie True ssi les coordonnées sont égales,
        sans prendre en compte la signature de l'objet.
        """
        if not isinstance(other, Point):
            return False
        if np.array_equal(self.coord, other.coord):
            return True
        elif np.allclose(self.coord, other.coord):
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    # NOTE Adapter à la 3D ?
    def plot2D(self, color="red", size=5):
        plt.plot(self.coord[0], self.coord[1], marker="o", markersize=size, color=color)

    def add_gmsh(self):
        """ Add the point to the current gmsh model via the python API."""
        if self.tag:
            # CAVEAT : True if the geometrical entity has already been instantiated.
            return None  # for information purposes only.
        self.tag = factory.addPoint(*self.coord, self.mesh_size)
        return self.tag



