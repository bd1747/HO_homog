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


def get_MeshFunction_val(msh_fctn):
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
    return (nb, val)


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


def function_from_xdmf(function_space, function_name, xdmf_path):
    """Read a finite element function from a xdmf file with checkpoint format.

    Parameters
    ----------
    function_space : dolfin.FunctionSpace
        Function space appropriate for the previously saved function.
        The mesh must be identical to the one used for the saved function.
    function_name : str
        The name of the saved function.
    xdmf_path : pathlib.Path
        Path of the xdmf file. It can be a Path object or a string.
        Extension must be '.xmdf'

    Returns
    -------
    dolfin.Function
        New function that is identical to the previously saved function.
    """
    f = fe.Function(function_space)
    file_in = fe.XDMFFile(str(xdmf_path))
    file_in.read_checkpoint(f, function_name)
    file_in.close()
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


def local_project(v, fspace, solver_method: str = "", **kwargs):
    """
    Info : https://comet-fenics.readthedocs.io/en/latest/tips_and_tricks.html#efficient-projection-on-dg-or-quadrature-spaces #noqa

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
    """
    metadata = kwargs.get("metadata", {})
    dX = kwargs.get("dx", fe.dx(metadata=metadata))

    dv = fe.TrialFunction(fspace)
    v_ = fe.TestFunction(fspace)
    a_proj = fe.inner(dv, v_) * dX
    b_proj = fe.inner(v, v_) * dX

    if solver_method == "LU":
        solver = fe.LocalSolver(
            a_proj, b_proj, solver_type=fe.cpp.fem.LocalSolver.SolverType.LU
        )
    elif solver_method == "Cholesky":
        solver = fe.LocalSolver(
            a_proj, b_proj, solver_type=fe.cpp.fem.LocalSolver.SolverType.Cholesky
        )
    else:
        solver = fe.LocalSolver(a_proj, b_proj)
    solver.factorize()
    u = fe.Function(fspace)
    solver.solve_local_rhs(u)
    return u


def _wrap_in_list(obj, name, types=type):
    """
    Transform single object or a collection of objects into a list.

    Source
    ------
    python/dolfin/fem/assembling.py, commit 4c72333
    """

    if obj is None:
        lst = []
    elif hasattr(obj, "__iter__"):
        lst = list(obj)
    else:
        lst = [obj]
    for obj in lst:
        if not isinstance(obj, types):
            raise TypeError(
                "expected a (list of) %s as '%s' argument" % (str(types), name)
            )
    return lst


def xdmf_mesh(mesh_file, import_subdomains=False, facet_file="", physical_file=""):
    """Create a FeniCS mesh from a mesh file with xdmf format.

    Parameters
    ----------
    mesh_file : str or Path
        Path of the file mesh (xdmf file)
    import_subdomains : bool, optional
        True if information about subdomains have to be imported.
        The paths of the auxiliary files that contains information about subdomains
        can be indicated with facet_file and physical_file.
        The paths used by default are :
            - "<mesh path>_facet_region.xdmf" and
            - "<mesh path>_physical_region.xdmf" (for subdomains)
    facet_file : str or Path, optional
        Path to the mesh auxiliary file that contains subdomains data.
        Defaults to "" i.e. the default path will be used.
    physical_file : str or Path, optional
        Path to the mesh auxiliary file that contains facet regions data.
        Defaults to "" i.e. the default path will be used.

    Returns
    -------
    Mesh / tuple
        If subdomains are not requested :
            - The Mesh instance
        If subdomains are requested :
            - The Mesh instance,
            - The MeshFunction for subdomains if it exists else None,
            - The MeshFunction for facets if it exists else None,

    Source
    ------
    Gist meshtagging_mvc.py, June 2018, Michal Habera
    https://gist.github.com/michalhabera/bbe8a17f788192e53fd758a67cbf3bed
    """
    if not isinstance(mesh_file, PurePath):
        mesh_file = Path(mesh_file)
    if not mesh_file.suffix == ".xdmf":
        raise TypeError("Wrong suffix for the path to the mesh.")
    mesh = fe.Mesh()
    with fe.XDMFFile(str(mesh_file)) as f_in:
        f_in.read(mesh)

    if not import_subdomains:
        return mesh

    dim = mesh.geometric_dimension()

    if not facet_file:
        facet_file = mesh_file.with_name(f"{mesh_file.stem}_facet_region.xdmf")
    else:
        facet_file = Path(facet_file)
        if not facet_file.suffix == ".xdmf":
            raise TypeError("Wrong suffix for the path to facet regions.")
    if not physical_file:
        physical_file = mesh_file.with_name(f"{mesh_file.stem}_physical_region.xdmf")
    else:
        physical_file = Path(physical_file)
        if not physical_file.suffix == ".xdmf":
            raise TypeError("Wrong suffix for the path to subdomains.")

    if not facet_file.exists():
        facet_regions = None
    else:
        facet_vc = fe.MeshValueCollection("size_t", mesh, dim - 1)
        with fe.XDMFFile(str(facet_file)) as f_in:
            f_in.read(facet_vc, "facet_data")
        facet_regions = fe.cpp.mesh.MeshFunctionSizet(mesh, facet_vc)
    if not physical_file.exists():
        subdomains = None
    else:
        cell_vc = fe.MeshValueCollection("size_t", mesh, dim - 1)
        with fe.XDMFFile(str(physical_file)) as f_in:
            f_in.read(cell_vc, "cell_data")
        subdomains = fe.cpp.mesh.MeshFunctionSizet(mesh, cell_vc)

    return mesh, subdomains, facet_regions

