# coding: utf8
"""
Created on 24/06/2019
@author: baptiste

Geometry toolbox,
affine transformations on Points, Lines, LineLoops...

Some features come from the transformations packages by Christoph Gohlke.
https://pypi.org/project/transformations/
Copyright :
Copyright (c) 2006-2019, Christoph Gohlke
Copyright (c) 2006-2019, The Regents of the University of California
Produced at the Laboratory for Fluorescence Dynamics
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice,
  this list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
from . import np, math, logger
from .point import Point
from .curves import Curve, Line, Arc
from .surfaces import LineLoop
from .tools import unit_vect


def geo_transformation_factory(pt_coord_fctn):
    """
    A partir d'une fonction élémentaire qui indique comment
    les coordonnées d'un point sont transformées,
    cette factory renvoie une fonction capable d'appliquer cette transformation
    sur une instance de Point, Line, Arc ou LineLoop.

    """

    def transformation(geo_ent, *args, **kwargs):
        """
        Renvoie un nouvel objet géométrique obtenu en appliquant la transformation
         à l'objet et tout ses parents.

        geo_ent : instance of a geometrical class Point, Curve and its subclasses
        or LineLoop

        """

        if geo_ent.tag:
            raise NotImplementedError(
                "For now a geometrical properties of a geometrical entity"
                "cannot be modified if this entity has already been added"
                "to the gmsh geometry model."
            )
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
            # TODO : rajouter le déterminant pour réorienter les lineloop
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


point_reflection = geo_transformation_factory(pt_reflection_basis)


def pln_reflection_basis(pt_coord, pln_pt, pln_normal):
    """
    Symétrie mirroir, en 3D, par rapport à un plan.
    Le plan est défini par une normale et un point contenu dans ce plan.
    """
    # ? laisser la possibilité d'utiliser directement un np.array ou une liste
    # ? pour donner les coordonnées pour plus de facilité d'utilisation ?
    if isinstance(pln_pt, Point):
        pln_pt = pln_pt.coord
    else:
        pln_pt = np.asarray(pln_pt)
        # Array interpretation of a. No copy is performed if it's already an ndarray
    n = unit_vect(pln_normal)
    return (pt_coord - pln_pt) - 2 * (pt_coord - pln_pt).dot(n) * n + pln_pt


plane_reflection = geo_transformation_factory(pln_reflection_basis)


def translation_basis(pt_coord, vect):
    """
    vect : 1-D array-like
        The 3 components of the vector that entirely define the translation.
    """
    vect = np.asarray(vect)
    return pt_coord + vect


translation = geo_transformation_factory(translation_basis)


def rotation_matrix(angle, direction, point=None):
    """
    Return matrix to rotate about axis defined by point and direction.

    Returns
    -------
    4×4 numpy.array
        Matrix representation of the rotation.

    From : transformations.py, https://www.lfd.uci.edu/~gohlke/

    >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
    >>> numpy.allclose(numpy.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
    True
    >>> angle = (random.random() - 0.5) * (2*math.pi)
    >>> direc = numpy.random.random(3) - 0.5
    >>> point = numpy.random.random(3) - 0.5
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> R0 = rotation_matrix(angle, direc, point)
    >>> R1 = rotation_matrix(-angle, -direc, point)
    >>> is_same_transform(R0, R1)
    True
    >>> I = numpy.identity(4, numpy.float64)
    >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
    True
    >>> numpy.allclose(2, numpy.trace(rotation_matrix(math.pi/2, direc, point)))
    True
    """

    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction = unit_vect(direction[:3])
    # rotation matrix around unit vector
    R = np.diag([cosa, cosa, cosa])
    R += np.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += np.array(
        [
            [0.0, -direction[2], direction[1]],
            [direction[2], 0.0, -direction[0]],
            [-direction[1], direction[0], 0.0],
        ]
    )
    M = np.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = np.array(point[:3], dtype=np.float64, copy=False)
        M[:3, 3] = point - np.dot(R, point)
    logger.debug(M)
    return M


def rotation_basis(pt_coord, angle, direction, point=None):
    """Apply a rotation operation to a point.
    The rotation axis is defined by a point and a direction.

    Parameters
    ----------
    pt_coord : 1-D array-like
        3 coordinates of the point to which the rotation is applied.
    angle : float
        Rotation angle
    direction : 1-D array
        Direction of the axis of rotation, 3 component vector
    point : 1-D array, optional
        3 oordinates of a point through which the axis of rotation passes.
        The default is None, which implies a rotation about the origin (0.,0.,0.)

    Returns
    -------
    rot_coord
        3 coordinates of the rotated point.

    >>> v1 = np.array((0.,1.,0.))
    >>> v3 = rotation_basis(v1, math.pi/2, [0, 0, 1], [1, 0, 0]))
    >>> np.allclose(np.array((0.,-1.,0.)),v3))
    True
    """

    R = rotation_matrix(angle, direction, point)
    x = np.pad(pt_coord, (0, 1), "constant", constant_values=(1.0))
    y = np.dot(R, x)
    rot_coord = y[:3]
    return rot_coord


rotation = geo_transformation_factory(rotation_basis)
# TODO : tester cette fonction
