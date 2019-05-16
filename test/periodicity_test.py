# coding: utf-8
"""
Created on 16/05/2019
@author: baptiste
"""

from ho_homog import periodicity
import numpy as np
import dolfin as fe
import logging
from pytest import approx

logger = logging.getLogger("Test_periodicity")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s',
    "%H:%M:%S")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def test_pbc_from_vectors():
    per_vect = np.array([[4.0, 0.0], [0.0, 8.0]])
    pbc = periodicity.PeriodicDomain.pbc_dual_base(per_vect, 'XY')
    test_points = [
        (0., 0.), (4., 0.), (4., 8.), (0., 8.),
        (2., 0.), (4., 4.), (2., 8.), (0., 4.),
    ]
    test_points = [np.array(coord) for coord in test_points]
    inside_results = [
        True, False, False, False,
        True, False, False, True,
    ]
    map_results = [
        (39996, 79992), (0., 0.), (0., 0.), (0., 0.),
        (39996, 79992), (0., 4.), (2., 0.), (39996, 79992),
    ]
    map_results = [np.array(coord) for coord in map_results]
    for point, result in zip(test_points, inside_results):
        logger.debug(f"Point : {point}, master : {pbc.inside(point, True)}")
        assert pbc.inside(point, on_boundary=True) == result

    y = np.zeros(2)
    for point, map_pt in zip(test_points, map_results):
        pbc.map(point, y)
        logger.debug(f"Point : {point}, map pt : {y}, expected : {map_pt}")
        assert y == approx(map_pt)


test_pbc_from_vectors()
