# coding: utf-8
"""
Created on 13/06/2019
@author: Baptiste Durand, baptiste.durand@enpc.fr

Collection of tools designed to help users working with gmsh python API.

"""

import logging

gmsh_logger = logging.getLogger("ho_homog.gmsh")


def process_gmsh_log(gmsh_log: list, detect_error=True):
    """Treatment of log messages emitted by the gmsh API."""
    err_msg, warn_msg = list(), list()
    for line in gmsh_log:
        if "error" in line.lower():
            err_msg.append(line)
        if "warning" in line.lower():
            warn_msg.append(line)
    gmsh_logger.info("**********")
    gmsh_logger.info(
        f"{len(gmsh_log)} logging messages from Gmsh : "
        f"{len(err_msg)} errors, {len(warn_msg)} warnings."
    )
    if err_msg:
        gmsh_logger.error("Gmsh errors details :")
        for line in err_msg:
            gmsh_logger.error(line)
    if warn_msg:
        gmsh_logger.warning("Gmsh warnings details :")
        for line in warn_msg:
            gmsh_logger.warning(line)
    gmsh_logger.debug("All gmsh messages :")
    gmsh_logger.debug(gmsh_log)
    gmsh_logger.info("**********")
    if detect_error and err_msg:
        raise AssertionError("Gmsh logging messages signal errors.")
