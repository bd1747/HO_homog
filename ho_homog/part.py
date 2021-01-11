# coding: utf8
"""
Created on 17/11/2018
@author: baptiste

"""

import logging
from pathlib import Path

import dolfin as fe
import matplotlib.pyplot as plt
import numpy as np

import ho_homog.materials as mat
import ho_homog.toolbox_FEniCS as fetools
from ho_homog.toolbox_gmsh import msh_conversion

logger = logging.getLogger(__name__)  # http://sametmax.com/ecrire-des-logs-en-python/


class FenicsPart:
    """
    Contrat : Créer un couple maillage + matériaux pour des géométries 2D, planes.
    """

    def __init__(
        self,
        mesh,
        materials,
        subdomains,
        global_dimensions=None,
        facet_regions=None,
        **kwargs,
    ):
        """
        Parameters
        ----------
        subdomains : dolfin.MeshFunction
            Indicates the subdomains that have been defined in the mesh.
        """
        self.mesh = mesh
        self.dim = mesh.topology().dim()  # dimension d'espace de depart
        self.materials = materials
        self.subdomains = subdomains
        self.global_dimensions = global_dimensions
        self.facet_regions = facet_regions
        if isinstance(materials, mat.Material):
            self.elasticity_tensor = fe.as_matrix(materials.get_C())
        elif isinstance(materials, dict):
            self.elasticity_tensor = mat.mat_per_subdomains(
                subdomains, materials, self.dim
            )
        else:
            raise TypeError("Expected a (dict of) Material as materials argument")
        if "bottom_left_corner" in kwargs.keys():
            self.bottom_left_corner = np.asarray(kwargs["bottom_left_corner"])
        self.name = kwargs.get("name", None)
        self._mat_area = None

    @property
    def mat_area(self):
        # Pattern : "lazy property" https://stevenloria.com/lazy-properties/
        # http://sametmax.com/creer-un-decorateur-a-la-volee/
        if self._mat_area is None:
            self._mat_area = fe.assemble(fe.Constant(1) * fe.dx(self.mesh))
        return self._mat_area

    @property
    def global_area(self):
        if self.global_dimensions is not None:
            return np.linalg.det(self.global_dimensions)
        else:
            msg = f"global_size information is lacking for FenicsPart {self}."
            raise AttributeError(msg)

    def plot_mesh(self):
        """ Plot the mesh of a FenicsPart in a matplotlib Figure."""
        plt.figure()
        try:
            fe.plot(self.mesh, title=f"Mesh of {self.name}")
        except AttributeError:
            fe.plot(self.mesh)

    def plot_subdomains(self):
        plt.figure()
        try:
            title = f"Subdomains imported for {self.name}"
        except AttributeError:
            title = "Imported subdomains"
        subdo_plt = fe.plot(self.subdomains, self.mesh, title=title)
        plt.colorbar(subdo_plt)

    def plot_facet_regions(self):
        plt.figure()
        facets_val = fetools.get_mesh_function_value(self.facet_regions)
        facets_val_range = max(facets_val[1]) - min(facets_val[1])
        cmap = plt.cm.get_cmap("viridis", facets_val_range)
        facets_plt = fetools.facet_plot2d(self.facet_regions, self.mesh, cmap=cmap)
        plt.colorbar(facets_plt[0])

    @staticmethod
    def part_from_file(
        mesh_path,
        materials,
        global_dimensions=None,
        subdomains_import=False,
        plots=False,
        **kwargs,
    ):
        """Generate an instance of FenicsPart from a .xml or .msh file that contains the mesh.

        Parameters
        ----------
        mesh_path : string or Path
            Relative or absolute path to the mesh file (format Dolfin XML, XDMF or MSH version 2)
        global_dimensions : 2D-array
            shape : 2×2 if 2D problem
            dimensions of the RVE
        materials : dict or Material
            [description] #TODO : à compléter
        subdomains_import : bool
            Import subdomains data ?
        plots : bool, optional
            If True (default) the physical regions and the facet regions are plotted at the end of the import.

        Returns
        -------
        FenicsPart instance

        """
        mesh_path = Path(mesh_path)
        name = mesh_path.stem
        suffix = mesh_path.suffix

        if suffix not in fetools.SUPPORTED_MESH_SUFFIX:
            mesh_file_paths = msh_conversion(
                mesh_path, format_=".xml", subdomains=subdomains_import
            )
            try:
                mesh_path = mesh_file_paths[0]
            except IndexError as error:
                mesh_path = mesh_file_paths
                logger.warning(error)
            suffix = mesh_path.suffix

        # Each supported mesh format -> one if structure
        subdomains, facets = None, None
        if suffix == ".xml":
            mesh = fe.Mesh(str(mesh_path))
            if subdomains_import:
                subdomains, facets = fetools.import_subdomain_data_xml(mesh, mesh_path)
        elif suffix == ".xdmf":
            mesh = fetools.xdmf_mesh(mesh_path)
            if subdomains_import:
                subdomains, facets = fetools.import_subdomain_data_xdmf(mesh, mesh_path)
                msg = f"Import of a mesh from {mesh_path} file, with subdomains data"
            else:
                msg = f"Import of a mesh from {mesh_path} file, without subdomains data"
            logger.info(msg)
        else:
            raise ValueError(f"expected a mesh path with a suffix `.xml` or `.xdmf`.")
        logger.info(f"Import of the mesh : DONE")

        if "name" not in kwargs:
            kwargs["name"] = name
        part = FenicsPart(
            mesh, materials, subdomains, global_dimensions, facets, **kwargs
        )
        if plots:
            part.plot_mesh()
            if subdomains is not None:
                part.plot_subdomains()
            if facets is not None:
                part.plot_facet_regions()
            plt.show()

        return part


class Fenics2DRVE(FenicsPart):
    """
    Create a mesh + constituent material(s) pair that represents a 2D RVE.
    This class is intended to be used as an input for 2D homogenization schemes, such as Fenics2DHomogenization.

    The RVE can  be made up of one constitutive material or several constitutive materials.
    In the second case, the distribution of materials must be indicated with a dolfin.MeshFunction.

    The RVE can contains one or several unit cells.
    """

    def __init__(
        self, mesh, generating_vectors, materials, subdomains, facet_regions, **kwargs,
    ):
        """
        Parameters
        ----------
        subdomains : dolfin.MeshFunction
            Indicates the subdomains that have been defined in the mesh.
        #! L'ordre des facet function à probablement de l'importance pour la suite des opérations
        """

        super().__init__(
            mesh, materials, subdomains, generating_vectors, facet_regions, **kwargs
        )
        self.gen_vect = generating_vectors
        self.rve_area = self.global_area
        self.C_per = self.elasticity_tensor

        self.epsilon = mat.epsilon

    def sigma(self, eps):
        return mat.sigma(self.C_per, eps)

    def strain_cross_energy(self, sig, eps):
        return mat.strain_cross_energy(sig, eps, self.mesh, self.rve_area)

    @staticmethod
    def rve_from_gmsh2drve(gmsh_2d_rve, materials, plots=True):
        """
        Generate an instance of Fenics2DRVE from a instance of the Gmsh2DRVE class.

        """
        msh_conversion(gmsh_2d_rve.mesh_abs_path, format_=".xml")
        mesh_path = gmsh_2d_rve.mesh_abs_path.with_suffix(".xml")
        gen_vect = gmsh_2d_rve.gen_vect
        kwargs = dict()
        try:
            kwargs["bottom_left_corner"] = gmsh_2d_rve.bottom_left_corner
        except AttributeError:
            pass
        fenics_rve = Fenics2DRVE.rve_from_file(
            mesh_path, gen_vect, materials, plots, **kwargs
        )
        return fenics_rve

    @staticmethod
    def rve_from_file(
        mesh_path, generating_vectors, materials, plots=True, **kwargs
    ):
        """Generate an instance of Fenics2DRVE from a .xml, .msh or .xdmf
        file that contains the mesh.

        Parameters
        ----------
        mesh_path : string or Path
            Relative or absolute path to the mesh file
            (format Dolfin XML, XDMF or MSH version 2)
        generating_vectors : 2D-array
            dimensions of the RVE
        materials : dictionnary
            [description] #TODO : à compléter
        plots : bool, optional
            If True (default) the physical regions and the facet regions are plotted
            at the end of the import.

        Returns
        -------
        Fenics2DRVE instance

        """
        if not isinstance(mesh_path, Path):
            mesh_path = Path(mesh_path)
        name = mesh_path.stem

        if mesh_path.suffix == ".xml":
            mesh = fe.Mesh(str(mesh_path))
        elif mesh_path.suffix == ".xdmf":
            mesh = fe.Mesh()
            with fe.XDMFFile(str(mesh_path)) as file_in:
                file_in.read(mesh)
        else:
            msh_conversion(mesh_path, format_=".xml")
            mesh_path = mesh_path.with_suffix(".xml")
            mesh = fe.Mesh(str(mesh_path))

        logger.info("Import of the mesh : DONE")
        # TODO : import des subdomains
        subdomains, facets = fetools.import_subdomain_data_xml(mesh, mesh_path)
        logger.info("Import of the subdomain data and facet-region data: DONE")

        if "name" not in kwargs:
            kwargs["name"] = name

        fenics_rve = Fenics2DRVE(
            mesh, generating_vectors, materials, subdomains, facets, **kwargs
        )

        if plots:
            fenics_rve.plot_mesh()
            if subdomains is not None:
                fenics_rve.plot_subdomains()
            if facets is not None:
                fenics_rve.plot_facet_regions()
            plt.show()

        return fenics_rve
