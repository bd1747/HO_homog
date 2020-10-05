# coding: utf-8
"""
Created on 16/05/2019
@author: baptiste
"""

from ho_homog import periodicity
import numpy as np
import logging
from pytest import approx

logger = logging.getLogger("Test_periodicity")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s", "%H:%M:%S"
)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def test_pbc_from_vectors():
    per_vect = np.array([[4.0, 0.0], [0.0, 8.0]])
    pbc = periodicity.PeriodicDomain.pbc_dual_base(per_vect, "XY")
    test_points = [
        (0.0, 0.0),
        (4.0, 0.0),
        (4.0, 8.0),
        (0.0, 8.0),
        (2.0, 0.0),
        (4.0, 4.0),
        (2.0, 8.0),
        (0.0, 4.0),
    ]
    test_points = [np.array(coord) for coord in test_points]
    inside_results = [
        True,
        False,
        False,
        False,
        True,
        False,
        False,
        True,
    ]
    map_results = [
        (39996, 79992),
        (0.0, 0.0),
        (0.0, 0.0),
        (0.0, 0.0),
        (39996, 79992),
        (0.0, 4.0),
        (2.0, 0.0),
        (39996, 79992),
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


def test_pbc_from_vectors_parallelogram():
    per_vect = np.array([[4.0, 2.0], [0.0, 4.0]])
    pbc = periodicity.PeriodicDomain.pbc_dual_base(per_vect, "XY")
    test_points = [
        (0, 0),
        (2, 0),
        (4, 0),
        (1, 2),
        (5, 2),
        (2, 4),
        (4, 4),
        (6, 4),
    ]
    test_points = [np.array(coord, np.float64) for coord in test_points]
    inside_results = [
        True,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
    ]
    for point, result in zip(test_points, inside_results):
        logger.debug(f"Point : {point}, master : {pbc.inside(point, True)}")
        assert pbc.inside(point, on_boundary=True) == result

    map_results = [
        (59994, 39996),
        (59994, 39996),
        (0, 0),
        (59994, 39996),
        (1, 2),
        (0, 0),
        (2, 0),
        (0, 0),
    ]
    map_results = [np.array(coord, np.float64) for coord in map_results]
    y = np.zeros(2)
    for point, map_pt in zip(test_points, map_results):
        pbc.map(point, y)
        logger.debug(f"Point : {point}, map pt : {y}, expected : {map_pt}")
        assert y == approx(map_pt)
    # * OK, test réussi
