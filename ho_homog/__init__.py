# coding: utf8

"""
#TODO : Décrire le package
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

__version__ = "0.1"


log_level = logging.DEBUG
log_path = Path("~/ho_homog_log/activity.log").expanduser()
if not log_path.parent.exists():
    log_path.parent.mkdir(mode=0o777, parents=True)
pckg_logger = logging.getLogger(__name__)
pckg_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    "%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s", "%Y-%m-%d %H:%M:%S"
)


GEO_TOLERANCE = 1e-12

try:
    from . import geometry, mesh_tools, mesh_generate
except ImportError:
    pckg_logger.warning("Import of geometry and mesh generating submodules fails")

try:
    from . import materials
except ImportError:
    pckg_logger.warning("Import of material submodule fails")

try:
    from . import periodicity
except ImportError:
    pckg_logger.warning("Import of periodicity submodule fails")


try:
    from . import full_scale_pb, homog2d, part, toolbox_FEniCS
except ImportError:
    pckg_logger.warning("One of the following submodule imports fails : full_scale_pb, homog2d, part, toolbox_FEniCS")




def set_log_handlers(level: int = log_level, path=log_path):
    """Should be run if log_path or log_level is changed."""
    file_handler = RotatingFileHandler(str(log_path), "a", 10000000, 10)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    pckg_logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    pckg_logger.addHandler(stream_handler)


def set_log_path(path):
    log_path = path
    for hdlr in pckg_logger.handlers[:]:
        pckg_logger.removeHandler(hdlr)
    set_log_handlers(log_level, log_path)


set_log_handlers(log_level, log_path)

__all__ = [
    "geometry",
    "homog2d",
    "materials",
    "mesh_generate_2D",
    "mesh_tools",
    "part",
    "full_scale_pb",
    "toolbox_FEniCS",
    "toolbox_gmsh" "set_log_path",
    "set_log_handlers",
    "log_level",
    "log_path",
    "pckg_logger",
    "GEO_TOLERANCE",
]
