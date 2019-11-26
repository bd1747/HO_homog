# coding: utf-8
"""
Created on 13/06/2019
@author: Baptiste Durand, baptiste.durand@enpc.fr

Collection of tools designed to help users working with gmsh python API.

"""

import logging
from pathlib import Path
import meshio
from subprocess import run

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


def conversion_to_xdmf(i_path, o_path, cell_reg, facet_reg, dim, subdomains=False):
    """Convert a ".msh" mesh generated with Gmsh to a xdmf mesh file.

    Parameters
    ----------
    i_path : Path
    o_path : Path
        Desired path for the converted mesh
    cell_reg : Path
        Desired path of the extra file for cell regions if subdomains are retained.
    facet_reg : Path
        Desired path of the extra file for facet regions if subdomains are retained.
    dim: int
        Geometrical dimension of the mesh (2D or 3D).
    subdomains : bool, optional
        If True, extra files are created to store information about subdomains.
        (default: False)
    """
    m = meshio.read(str(i_path))
    if dim == 2:
        m.points = m.points[:, :2]
        geo_only = meshio.Mesh(points=m.points, cells={"triangle": m.cells["triangle"]})
        cell = "triangle"
        face = "line"
    elif dim == 3:
        raise NotImplementedError("3D meshes are not supported yet.")
    else:
        ValueError
    meshio.write(str(o_path), geo_only)
    if subdomains:
        cell_funct = meshio.Mesh(
            points=m.points,
            cells={cell: m.cells[cell]},
            cell_data={cell: {"cell_data": m.cell_data[cell]["gmsh:physical"]}},
        )
        meshio.write(str(cell_reg), cell_funct)
        facet_funct = meshio.Mesh(
            points=m.points,
            cells={face: m.cells[face]},
            cell_data={face: {"facet_data": m.cell_data[face]["gmsh:physical"]}},
        )
        meshio.write(str(facet_reg), facet_funct)
    return True


def msh_conversion(
    mesh,
    format_: str = ".xdmf",
    output_dir=None,
    subdomains: bool = False,
    dim: int = 2,
):
    """
    Convert a ".msh" mesh generated with Gmsh to a format suitable for FEniCS.

    Parameters
    ----------
    mesh : Path or str
        Path that points to the existing mesh file.
    format : str
        Suffix desired for the mesh file. (default: ".xdmf")
        Supported suffixes :
            - ".xdmf"
            - ".xml" (DOLFIN xml format)
    output_dir : Path, optional
        Path of the directory where the converted mesh file must be written.
        If None, the converted file is written in the same directory
        as the input file. (default: None)
    subdomains : bool, optional
        If True, extra files are created to store information about subdomains.
        (default: False)
    dim: int, optional
        Geometrical dimension of the mesh (2D or 3D). (default: 2)

    Returns
    -------
    Path / tuple
    If subdomain conversion is not requested :
        - Path to the mesh file
    If subdomain conversion is requested :
        - Path to the mesh file,
        - Path to the extra files for subdomains if it exists else None,
        - Path to the extra files for facet regions if it exists else None,

    Warning
    -------
    A specific version of the MSH format should be use in accordance with the
    desired output format :
        - ".xml" output -> MSH file format version 2;
        - ".xdmf" output -> MSH file format version 4.
    """
    input_path = Path(mesh)
    name = input_path.stem

    if format_ not in (".xml", ".xdmf"):
        raise TypeError
    mesh_path = input_path.with_suffix(format_)
    if output_dir:
        mesh_path = output_dir.joinpath(mesh_path.name)
    physical_region = mesh_path.with_name(name + "_physical_region" + format_)
    facet_region = mesh_path.with_name(name + "_facet_region" + format_)
    if physical_region.exists():
        physical_region.unlink()
    if facet_region.exists():
        facet_region.unlink()

    if format_ == ".xml":
        cmd = f"dolfin-convert {input_path} {mesh_path}"
        run(cmd, shell=True, check=True)
    elif format_ == ".xdmf":
        conversion_to_xdmf(
            input_path, mesh_path, physical_region, facet_region, dim, subdomains
        )
    if subdomains:
        return (
            mesh_path,
            physical_region if physical_region.exists() else None,
            facet_region if facet_region.exists() else None,
        )
    else:
        return mesh_path
