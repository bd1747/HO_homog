# coding: utf-8
"""
Created on 17/01/2019
@author: Baptiste Durand, baptiste.durand@enpc.fr

Collection of tools designed to help users working with FEniCS objects.

"""
import logging

import dolfin as fe
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path, PurePath

logger = logging.getLogger(__name__)

DOLFIN_KRYLOV_METHODS = {
    "bicgstab": "Biconjugate gradient stabilized method",
    "cg": "Conjugate gradient method",
    "default": "default Krylov method",
    "gmres": "Generalized minimal residual method",
    "minres": "Minimal residual method",
    "richardson": "Richardson method",
    "tfqmr": "Transpose-free quasi-minimal residual method",
}

DOLFIN_LU_METHODS = {
    "default": "default LU solver",
    "mumps": "MUMPS (MUltifrontal Massively Parallel Sparse direct Solver)",
    "petsc": "PETSc built in LU solver",
    "superlu": "SuperLU",
    "umfpack": "UMFPACK (Unsymmetric MultiFrontal sparse LU factorization)",
}


SUPPORTED_MESH_SUFFIX = (".xml", ".xdmf")


def get_mesh_function_value(msh_fctn):
    """
    Get information about the set of values that a given MeshFunction outputs.

    Parameters
    ------
    msh_fctn : instance of MeshFunction

    Returns
    -------
    Tuple (nb, vals)
    nb : int
        number of different values
    vals : list
        list of the different values that the MeshFunction outputs on its definition mesh.
    """

    val = np.unique(msh_fctn.array())
    nb = len(val)
    return nb, val


def facet_plot2d(
    facet_func, mesh, mesh_edges=True, markers=None, exclude_val=(0,), **kargs
):
    """
    Source : https://bitbucket.org/fenics-project/dolfin/issues/951/plotting-facetfunctions-in-matplotlib-not #noqa
    """
    x_list, y_list = [], []
    if markers is None:
        for facet in fe.facets(mesh):
            mp = facet.midpoint()
            x_list.append(mp.x())
            y_list.append(mp.y())
        values = facet_func.array()
    else:
        i = 0
        values = []
        for facet in fe.facets(mesh):
            if facet_func[i] in markers:
                mp = facet.midpoint()
                x_list.append(mp.x())
                y_list.append(mp.y())
                values.append(facet_func[i])
            i += 1
    if exclude_val:
        filtered_data = [], [], []
        for x, y, val in zip(x_list, y_list, values):
            if val in exclude_val:
                continue
            filtered_data[0].append(x)
            filtered_data[1].append(y)
            filtered_data[2].append(val)
        x_list, y_list, values = filtered_data

    plots = [plt.scatter(x_list, y_list, s=30, c=values, linewidths=1, **kargs)]
    if mesh_edges:
        plots.append(fe.plot(facet_func.mesh()))
    return plots


def function_from_xdmf(function_space, function_name="", xdmf_path=""):
    """Read a finite element function from a xdmf file with checkpoint format.
        Multiple functions can also be read at once.

    Parameters
    ----------
    function_space : dolfin.FunctionSpace
        Function space appropriate for the previously saved function.
        The mesh must be identical to the one used for the saved function.
    function_name : str or list of str
        The name(s) of the saved function.
    xdmf_path : pathlib.Path or str
        Path of the xdmf file. It can be a Path object or a string.
        Extension must be '.xmdf'

    If multiple functions are requested: #TODO : test
    the first argument must be a list of tuples (FunctionSpace, function name)
    the second argument must be empty.

    Returns
    -------
    dolfin.Function, list of dolfin.Function
        New function(s) that is(are) identical to the previously saved function(s).

    Example
    --------
    >>> import dolfin as fe
    >>> mesh = xdmf_mesh("mesh.xdmf")
    >>> V = fe.VectorFunctionSpace(mesh, "CG", 2)
    >>> W = fe.VectorFunctionSpace(mesh, "DG", 1)
    >>> function_from_xdmf([(V,"u"),(W,"eps")], xdmf_path="checkpoint.xdmf")
    """

    if isinstance(function_space, list):
        all_f = list()
        with fe.XDMFFile(str(xdmf_path)) as f_in:
            for fspace, fname in function_space:
                f = fe.Function(fspace)
                f_in.read_checkpoint(f, fname)
                f.rename(fname, "")
                all_f.append(f)
        return all_f
    else:
        f = fe.Function(function_space)
        f_in = fe.XDMFFile(str(xdmf_path))
        f_in.read_checkpoint(f, function_name)
        f_in.close()
        f.rename(function_name, "")
        return f


def function_errornorm(u, v, norm_type="L2", enable_diff_fspace=False):
    """Compute the difference between two functions
    defined on the same functionspace with the given norm.

    If the two objects are not functions defined in the same functionspace
    the FEniCS function errornorm should be used.
    Alternatively the constraint of function space equality can be relaxed
    with caution with the enable_diff_fspace flag.

    Based of the FEniCS functions errornorm and norm.

    Parameters
    ----------
    u, v : dolfin.functions.function.Function
    norm_type : string
        Type of norm. The :math:`L^2` -norm is default.
        For other norms, see :py:func:`norm <dolfin.fem.norms.norm>`.
    enable_diff_fspace: bool
        Relax the constraint of function space equality

    Returns
    -------
    float
        Norm of the difference
    """
    # TODO: Indiquer la différence avec la fonction errornorm de FEnics, si elle existe.
    # TODO : Quelle est la différence avec fe.errornorm ?
    # TODO : Dans Tutorial 1 FEniCS, il est dans le paragraphe sur errornorm que les 2 fonctions sont
    # interpolées dans un espace fonctionnel de degré supérieur pour avoir une meilleure précision.
    # Est-ce que je ne devrais pas faire ça ici, également ?
    if u.function_space() == v.function_space():
        difference = fe.Function(u.function_space())
        difference.assign(u)
        difference.vector().axpy(-1.0, v.vector())
    elif enable_diff_fspace:
        logger.warning("Function spaces not equals.")
        logger.warning(f"Projection to compute the difference between {u} and {v}")
        difference = fe.project(u - v, u.function_space())
    else:
        raise RuntimeError("Cannot compute error norm, Function spaces do not match.")
    return fe.norm(difference, norm_type)


_local_solvers = list()


def local_project(v, fspace, solver_method: str = "", **kwargs):
    """
    Parameters
    ----------
    v : [type]
        input field
    fspace : [type]
        [description]
    solver_method : str, optional
        "LU" or "Cholesky" factorization. LU method used by default.

    keyword arguments
    ----------
    metadata : dict, optional
        This can be used to deﬁne diﬀerent quadrature degrees for diﬀerent
        terms in a form, and to override other form compiler speciﬁc options
        separately for diﬀerent terms. By default {}
        See UFL user manual for more information
        ** WARNING** : May be over ignored if dx argument is also used. (Non étudié)
    dx : Measure
        Impose the dx measure for the projection.
        This is for example useful to do some kind of interpolation on DG functionspaces.

    Returns
    -------
    Function

    Notes
    -----
    Source for this code:
    https://comet-fenics.readthedocs.io/en/latest/tips_and_tricks.html#efficient-projection-on-dg-or-quadrature-spaces

    """
    metadata = kwargs.get("metadata", {})
    dx = kwargs.get("dx", fe.dx(metadata=metadata))

    reuse_solver = kwargs.get(
        "reuse_solver", True
    )  # Reuse a LocalSolver that already exists
    id_function_space = (
        fspace.id()
    )  # The id is unique. https://fenicsproject.org/olddocs/dolfin/latest/cpp/d8/df0/classdolfin_1_1Variable.html#details

    dv = fe.TrialFunction(fspace)
    v_ = fe.TestFunction(fspace)
    a_proj = fe.inner(dv, v_) * dx
    b_proj = fe.inner(v, v_) * dx

    if solver_method == "LU":
        solver_type = fe.cpp.fem.LocalSolver.SolverType.LU
        solver = fe.LocalSolver(a_proj, b_proj, solver_type=solver_type)
    elif solver_method == "Cholesky":
        solver_type = fe.cpp.fem.LocalSolver.SolverType.Cholesky
        solver = fe.LocalSolver(a_proj, b_proj, solver_type=solver_type)
    else:
        solver = fe.LocalSolver(a_proj, b_proj)

    if not (reuse_solver and ((id_function_space, solver_method) in _local_solvers)):
        solver.factorize()
        # You can optionally call the factorize() method to pre-calculate the local left-hand side factorizations to speed up repeated applications of the LocalSolver with the same LHS
        # This class can be used for post-processing solutions, e.g.computing stress fields for visualisation, far more cheaply that using global projections
        # https://fenicsproject.org/olddocs/dolfin/latest/cpp/de/d86/classdolfin_1_1LocalSolver.html#details
        _local_solvers.append((id_function_space, solver_method))

    u = fe.Function(fspace)
    solver.solve_local_rhs(u)
    return u


def xdmf_mesh(mesh_file):
    """Create a FeniCS mesh from a mesh file with xdmf format.

    Parameters
    ----------
    mesh_file : str or Path
        Path of the file mesh (xdmf file)

    Returns
    -------
    dolfin.Mesh
    """
    if not isinstance(mesh_file, PurePath):
        mesh_file = Path(mesh_file)
    if not mesh_file.suffix == ".xdmf":
        raise TypeError("Wrong suffix for the path to the mesh.")
    mesh = fe.Mesh()
    with fe.XDMFFile(str(mesh_file)) as f_in:
        f_in.read(mesh)
    return mesh


_warning_message_ = "The {} file is missing for the mesh {}."


def import_subdomain_data_xml(mesh, mesh_file):
    """Import data from the _physical_region.xml and _facet_region.xml files
    that result from the mesh conversion to .xml with dolfin-convert

    Parameters
    ----------
    mesh: dolfin.Mesh
        mesh already imported from the `.xml` file (mesh_path)
    mesh_file : pathlib.Path
        Path to the mesh file (.xml).
        The files that contain the data about the subdomains and the facet regions
        are assumed to be in the same folder, with the names:
        `[mesh_name]_physical_region.xml` and `[mesh_name]_facet_region.xml`.
    """
    name = mesh_file.stem
    subdo_path = mesh_file.with_name(name + "_physical_region.xml")
    facet_path = mesh_file.with_name(name + "_facet_region.xml")

    if subdo_path.exists():
        subdomains = fe.MeshFunction("size_t", mesh, subdo_path.as_posix())
        subdo_val = get_mesh_function_value(subdomains)
        logger.info(f"{subdo_val[0]} physical regions imported. Tags : {subdo_val[1]}")
    else:

        logger.warning(_warning_message_.format("_physical_region.xml", mesh_file.name))

        subdomains = None

    if facet_path.exists():
        facets = fe.MeshFunction("size_t", mesh, facet_path.as_posix())
        facets_val = get_mesh_function_value(facets)
        logger.info(f"{facets_val[0]} facet regions imported. Tags : {facets_val[1]}")
    else:
        logger.warning(_warning_message_.format("_facet_region.xml", mesh_file.name))

        facets = None
    return subdomains, facets


def import_subdomain_data_xdmf(mesh, mesh_file, facet_file="", physical_file=""):

    """Import information about subdomains from .xdmf files from
    obtained with the mesh conversion .msh -> .xdmf with meshio.

    The paths of the auxiliary files that contains information about subdomains
    can be indicated with facet_file and physical_file.
    The paths used by default are :
        - "<mesh path>_facet_region.xdmf" and
        - "<mesh path>_physical_region.xdmf" (for subdomains)


    Parameters
    ----------
    mesh: dolfin.Mesh
        mesh already imported from the `.xdmf` file (mesh_path)
    mesh_file : pathlib.Path
        Path to the mesh file (.xdmf).
    facet_file : str or Path, optional
        Path to the mesh auxiliary file that contains subdomains data.
        Defaults to "" i.e. the default path will be used.
    physical_file : str or Path, optional
        Path to the mesh auxiliary file that contains facet regions data.
        Defaults to "" i.e. the default path will be used.
    Returns
    -------
    Tuple (length: 2)
        - The MeshFunction for subdomains if it exists else None;
        - The MeshFunction for facets if it exists else None.

    Source
    ------
    Gist meshtagging_mvc.py, June 2018, Michal Habera
    https://gist.github.com/michalhabera/bbe8a17f788192e53fd758a67cbf3bed

    """

    dim = mesh.geometric_dimension()

    if not physical_file:
        physical_file = mesh_file.with_name(f"{mesh_file.stem}_physical_region.xdmf")
    if not facet_file:
        facet_file = mesh_file.with_name(f"{mesh_file.stem}_facet_region.xdmf")
    physical_file, facet_file = Path(physical_file), Path(facet_file)
    for p, n in zip((physical_file, facet_file), ("subdomains", "facet regions")):
        if not p.suffix == ".xdmf":
            raise TypeError(f"Wrong suffix for the path to {n}.")

    subdomains, facets = None, None
    if physical_file.exists():
        cell_vc = fe.MeshValueCollection("size_t", mesh, dim)
        with fe.XDMFFile(str(physical_file)) as f_in:
            f_in.read(cell_vc, "cell_data")
        subdomains = fe.cpp.mesh.MeshFunctionSizet(mesh, cell_vc)
    else:
        logger.warning(
            _warning_message_.format("_physical_region.xdmf", mesh_file.name)
        )

    if facet_file.exists():
        facet_vc = fe.MeshValueCollection("size_t", mesh, dim - 1)
        with fe.XDMFFile(str(facet_file)) as f_in:
            f_in.read(facet_vc, "facet_data")
        facets = fe.cpp.mesh.MeshFunctionSizet(mesh, facet_vc)
    else:
        logger.warning(_warning_message_.format("_facet_region.xdmf", mesh_file.name))
    return subdomains, facets
