# coding: utf-8
"""
Created on 13/06/2019
@author: Baptiste Durand, baptiste.durand@enpc.fr

Collection of tools designed to help users working with gmsh python API.

"""

import logging
from pathlib import Path
from . import pckg_logger

try:
    import meshio
except ImportError:
    pckg_logger.warning("Import of meshio fails.")

from subprocess import run
import gmsh

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

    Source
    ------
    https://fenicsproject.discourse.group/t/transitioning-from-mesh-xml-to-mesh-xdmf-from-dolfin-convert-to-meshio/412/79?u=bd1747 #noqa
    https://fenicsproject.discourse.group/t/transitioning-from-mesh-xml-to-mesh-xdmf-from-dolfin-convert-to-meshio/412/123?u=bd1747 #noqa

    """
    m = meshio.read(i_path)
    # points: 2Darray( point, xyz)
    # cells: list( cellblocks( type='line'/'triangle', data=2Darray(element, points) ))
    # cell_data: dict('gmsh:physical', 2Darray(geo_tag, ph_tag))
    # field_data: dict('top'/'bottom'/PH_NAME, 1Darray (ph_tag, geo_type)

    if dim == 2:
        m.points = m.points[:, :2]
        geo_only = meshio.Mesh(
            points=m.points, cells=[("triangle", m.cells_dict["triangle"])]
        )
        cell_key = "triangle"
        face_key = "line"

    elif dim == 3:
        raise NotImplementedError("3D meshes are not supported yet.")
        # * INFO :
    else:
        raise ValueError

    meshio.write(o_path, geo_only)

    if subdomains:
        # TODO : à tester !
        cell_data, facet_data = list(), list()
        for key, data in m.cell_data_dict["gmsh:physical"].items():
            if key == cell_key:
                cell_data.append(data)
            elif key == face_key:
                facet_data.append(data)

        cell_mesh, facet_mesh = list(), list()
        for cell in m.cells:
            if cell.type == cell_key:
                cell_mesh.append((cell_key, cell.data))
            if cell.type == face_key:
                facet_mesh.append((face_key, cell.data))

        cell_funct = meshio.Mesh(
            points=m.points, cells=cell_mesh, cell_data={"cell_data": cell_data},
        )
        facet_funct = meshio.Mesh(
            points=m.points, cells=facet_mesh, cell_data={"facet_data": facet_data},
        )
        # TODO : Regarder, si on se sert des sous-domaines de gmsh, si on peut mettre les cell data et les facet data dans le même fichier
        # TODO : c.à.d cell_data = {"cell_data": cell_data, "facet_data":facet_data}
        meshio.write(cell_reg, cell_funct)
        meshio.write(facet_reg, facet_funct)
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
