# coding: utf8
"""
Created on 08/04/2019
@author: baptiste
"""

import logging
import warnings
from pathlib import Path

import dolfin as fe
import numpy as np

import ho_homog.materials as mat
from ho_homog.toolbox_FEniCS import (
    DOLFIN_KRYLOV_METHODS,
    DOLFIN_LU_METHODS,
    local_project,
)

logger = logging.getLogger(__name__)

try:
    from fenicstools import interpolate_nonmatching_mesh_any
except ImportError:
    warnings.warn("Import of fenicstools has failed.", ImportWarning)
    logger.warning("Import : fenicstools cannot be imported.")


# * For mechanical fields reconstruction
MACRO_FIELDS_NAMES = ["U", "E", "EG", "EGG"]


class FullScaleModel(object):
    """Solve an elasticity problem over a 2D domain that contains a periodic microstructure or a homogeneous material or that has a more complex material behavior."""

    def __init__(self, fenics_2d_part, loads, boundary_conditions, element):
        """

        The FacetFunction must be the same for all BC and facet loads. Its type of value must be 'size_t' (unsigned integers).

        Parameters
        ----------
        fenics_2d_part : FEnicsPart
            [description]
        loads : [type]
            [description]
        boundary_conditions : list or tuple
            All the boundary conditions.
            Each of them is describe by a tuple or a dictionnary.
            Only one periodic BC can be prescribed.
            Format:
                if Periodic BC :
                    {'type': 'Periodic', 'constraint': PeriodicDomain}
                    or
                    ('Periodic', PeriodicDomain)
                if Dirichlet BC :
                    {
                        'type': 'Dirichlet',
                        'constraint': the prescribed value,
                        'facet_function': facet function,
                        'facet_idx': facet function index}
                    or
                    ('Dirichlet', the prescribed value, facet function, facet function index)
                    or
                    ('Dirichlet', the prescribed value, indicator function)
                        where the indicator function is python function like :
                        def clamped(x, on_boundary):
                            return on_boundary and x[0] < tolerance
        element: tuple or dict
        Ex: ('CG', 2) or {'family':'Lagrange', degree:2}
        """
        self.part = fenics_2d_part

        # * Boundary conditions
        self.per_bc = None
        self.Dirichlet_bc = list()
        for bc in boundary_conditions:
            if isinstance(bc, dict):
                bc_ = bc
                bc = [bc_["type"], bc_["constraint"]]
                try:
                    bc.append(bc_["facet_function"])
                    bc.append(bc_["facet_idx"])
                except KeyError:
                    pass
            bc_type, *bc_data = bc
            if bc_type == "Periodic":
                if self.per_bc is not None:
                    raise AttributeError(
                        "Only one periodic boundary condition can be prescribed."
                    )
                self.per_bc = bc_data[0]
            elif bc_type == "Dirichlet":
                if len(bc_data) == 2 or len(bc_data) == 3:
                    bc_data = tuple(bc_data)
                else:
                    raise AttributeError(
                        "Too much parameter for the definition of a Dirichlet BC."
                    )
                self.Dirichlet_bc.append(bc_data)

        self.measures = {self.part.dim: fe.dx, self.part.dim - 1: fe.ds}

        # * Function spaces
        try:
            self.elmt_family = family = element["family"]
            self.elmt_degree = degree = element["degree"]
        except TypeError:  # Which means that element is not a dictionnary
            self.element_family, self.element_degree = element
            family, degree = element
        cell = self.part.mesh.ufl_cell()
        Voigt_strain_dim = int(self.part.dim * (self.part.dim + 1) / 2)
        strain_deg = degree - 1 if degree >= 1 else 0
        strain_FE = fe.VectorElement("DG", cell, strain_deg, dim=Voigt_strain_dim)
        self.scalar_fspace = fe.FunctionSpace(
            self.part.mesh,
            fe.FiniteElement(family, cell, degree),
            constrained_domain=self.per_bc,
        )
        self.displ_fspace = fe.FunctionSpace(
            self.part.mesh,
            fe.VectorElement(family, cell, degree, dim=self.part.dim),
            constrained_domain=self.per_bc,
        )
        self.strain_fspace = fe.FunctionSpace(
            self.part.mesh, strain_FE, constrained_domain=self.per_bc
        )

        self.v = fe.TestFunction(self.displ_fspace)
        self.u = fe.TrialFunction(self.displ_fspace)
        self.a = (
            fe.inner(
                mat.sigma(self.part.elasticity_tensor, mat.epsilon(self.u)),
                mat.epsilon(self.v),
            )
            * self.measures[self.part.dim]
        )
        self.K = fe.assemble(self.a)

        # * Create suitable objects for Dirichlet boundary conditions
        for i, bc_data in enumerate(self.Dirichlet_bc):
            self.Dirichlet_bc[i] = fe.DirichletBC(self.displ_fspace, *bc_data)
        # ? Vu comment les conditions aux limites de Dirichlet interviennent dans le problème, pas sûr que ce soit nécessaire que toutes soient définies avec la même facetfunction

        # * Taking into account the loads
        if loads:
            self.set_loads(loads)
        else:
            self.loads_data = None
            self.load_integrals = None
        # * Prepare attribute for the solver
        self.solver = None

    def set_solver(self, solver_method="mumps", **kwargs):
        """
        Choose the type of the solver and its method.

        An up-to-date list of the available solvers and preconditioners
        can be obtained with dolfin.list_linear_solver_methods() and
        dolfin.list_krylov_solver_preconditioners().

        kwargs:
        type e.g. 'LU',
        preconditioner e.g. 'default'
        """
        s_type = kwargs.pop("type", None)
        s_precond = kwargs.pop("preconditioner", "default")

        if s_type is None:
            if solver_method in DOLFIN_KRYLOV_METHODS.keys():
                s_type = "Krylov"
            elif solver_method in DOLFIN_LU_METHODS.keys():
                s_type = "LU"
            else:
                raise RuntimeError("The indicated solver method is unknown.")
        else:
            if not (
                solver_method in DOLFIN_KRYLOV_METHODS.keys()
                or solver_method in DOLFIN_LU_METHODS.keys()
            ):
                raise RuntimeError("The indicated solver method is unknown.")
        self._solver = dict(type=s_type, method=solver_method, preconditioner=s_precond)
        if s_precond:
            self._solver["preconditioner"] = s_precond

        if s_type == "Krylov":
            self.solver = fe.KrylovSolver(self.K, solver_method)
        elif s_type == "LU":
            self.solver = fe.LUSolver(self.K, solver_method)
        if s_precond != "default":
            self.solver.parameters.preconditioner = s_precond
        return self.solver

    def set_loads(self, loads):
        """Define the loads of the elasticy problem.

        Parameters
        ----------
        loads : List
            List of all the loads of the problem.
            Each load must be described by a tuple.
            3 formats can be used to define a load :
                - Topology dimension, Expression valid throughout the domain
                - Topology dimension, Magnitude, distribution over the domain
                    -> load value at x = magnitude(x)*distribution(x)
                - Topology dimension, Magnitude, MeshFunction, subdomain_index

        Returns
        -------
        list
            The self.loads attribute
        """
        self.loads_data = dict()
        self.load_subdomains = dict()
        for topo_dim, *load_data in loads:
            try:
                self.loads_data[topo_dim].append(load_data)
            except KeyError:
                self.loads_data[topo_dim] = [load_data]
            if len(load_data) == 3:
                mesh_fctn = load_data[1]
                try:
                    if not mesh_fctn == self.load_subdomains[topo_dim]:
                        raise ValueError(
                            "Only one mesh function for each topology dimension"
                            " can be used for the definition of loads."
                        )
                except KeyError:
                    self.load_subdomains[topo_dim] = mesh_fctn

        # * Define load integrals
        labels = {self.part.dim: "dx", self.part.dim - 1: "ds"}
        for topo_dim, partition in self.load_subdomains.items():
            self.measures[topo_dim] = fe.Measure(
                labels[topo_dim], domain=self.part.mesh, subdomain_data=partition
            )

        self.load_integrals = dict()
        for topo_dim, loads in self.loads_data.items():
            self.load_integrals[topo_dim] = list()
            dy = self.measures[topo_dim]
            for load in loads:
                if len(load) == 1:
                    contrib = fe.dot(load[0], self.v) * dy
                elif len(load) == 2:
                    contrib = fe.dot(load[1] * load[0], self.v) * dy
                elif len(load) == 3:
                    contrib = fe.dot(load[0], self.v) * dy(load[2])
                else:
                    raise AttributeError(
                        "Too much parameter for the definition of a load. "
                        "Expecting 2, 3 or 4 parameters for each load."
                    )
                self.load_integrals[topo_dim].append(contrib)

        return self.loads_data

    def solve(self, results_file=None):
        if self.solver is None:
            logger.warning("The solver has to be set.")
        if self.load_integrals is not None:
            L_terms = []
            for contrib_list in self.load_integrals.values():
                L_terms += contrib_list
            L = sum(L_terms)
        else:
            zero = fe.Constant(np.zeros(shape=self.v.ufl_shape))
            L = fe.dot(zero, self.v) * self.measures[self.part.dim]

        logger.info("Assembling system...")
        K, res = fe.assemble_system(self.a, L, self.Dirichlet_bc)
        logger.info("Assembling system : done")
        self.u_sol = fe.Function(self.displ_fspace)
        logger.info("Solving system...")
        self.solver.solve(K, self.u_sol.vector(), res)
        logger.info("Solving system : done")
        logger.info("Computing strain solution...")
        eps = mat.epsilon(self.u_sol)
        self.eps_sol = local_project(eps, self.strain_fspace, solver_method="LU")
        logger.info("Saving results...")
        if results_file is not None:
            try:
                if results_file.suffix != ".xdmf":
                    results_file = results_file.with_suffix(".xdmf")
            except AttributeError:
                results_file = Path(results_file).with_suffix(".xdmf")
            with fe.XDMFFile(results_file.as_posix()) as ofile:
                ofile.parameters["flush_output"] = False
                ofile.parameters["functions_share_mesh"] = True
                self.u_sol.rename(
                    "displacement", "displacement solution, full scale problem"
                )
                self.eps_sol.rename("strain", "strain solution, full scale problem")
                ofile.write(self.u_sol, 0.0)
                ofile.write(self.eps_sol, 0.0)
        return self.u_sol


def reconstruction(
    localization_tensors: dict,
    macro_kinematic: dict,
    function_spaces: dict,
    localization_rules: dict = {},
    output_request=("u", "eps"),
    **kwargs,
):
    """
    One argument among localization_rules and trunc_order must be used.

    Parameters
    ----------
    localization_tensors : dictionnary
        Format : {
            [Macro_field] : {
                [micro_field] :
                    [list of lists : for each Macro_field component a list contains the components of the localization tensor field.]
            }
        }
    macro_kinematic : dictionnary of fields
        The macroscopic kinematic fields.
        Keys : 'U', 'EG', 'EGG',\\dots
        Values : lists of components.
        None can be used to indicate a component that is equal to 0 or irrelevant.
    function_spaces : dictionnary
        Function space into which each mechanical field have to be built.
            - keys : 'u', 'eps' or 'sigma'
            - values : FEniCS function space
    localization_rules : dict, optional
        The rules that have to be followed for the construction of the fluctuations.
        Defaults to {} i.e. the trunc_order parameter will be used.
    output_request : tuple of strings, optional
        Fields that have to be calculated.
        The request must be consistent with the keys of other parameters :
            - function_spaces
            - localization_rules
        outputs can be :
            =======  ===================
            name      Description
            =======  ===================
            'u'      displacement field
            'eps'    strain field
            'sigma'  stress field
            =======  ===================
        Defaults to ('u', 'eps').

    Return
    ------
    Dictionnary
        Mechanical fields with microscopic fluctuations.
        Keys are "eps", "sigma" and "u" respectively for the strain, stress and displacement fields.

    Other Parameters
    ----------------
    **kwargs :
        Valid kwargs are
            ===============  =====  =============================================
            Key              Type   Description
            ===============  =====  =============================================
            interp_fnct       str   The name of the desired function for the interpolations. Allowed values are : "dolfin.interpolate" and "interpolate_nonmatching_mesh_any"
            trunc_order       int    Order of truncation for the reconstruction of the displacement according to the notations used in ???. Override localization_rules parameter.
            ===============  ======  ============================================




    """
    # TODO : récupérer topo_dim à partir des tenseurs de localisation, ou mieux, à partir des espaces fonctionnels
    # TODO : choisir comment on fixe la len des listes correspondantes aux composantes de u et de epsilon.

    # TODO : permettre la translation du RVE par rapport à la structure macro autre part
    # TODO :  translation_microstructure: np.array, optional
    # TODO :         Vector, 1D array (shape (2,) or (3,)), position origin used for the description of the RVE with regards to the macroscopic origin.

    # Au choix, utilisation de trunc_order ou localization_rules dans les kargs
    trunc_order = kwargs.pop("trunc_order", None)
    if trunc_order:
        localization_rules = {
            "u": [
                (MACRO_FIELDS_NAMES[i], MACRO_FIELDS_NAMES[i])
                for i in range(trunc_order + 1)
            ],
            "eps": [
                (MACRO_FIELDS_NAMES[i], MACRO_FIELDS_NAMES[i])
                for i in range(1, trunc_order + 1)
            ],
        }
    # * Ex. for truncation order = 2:
    # * localization_rules = {
    # *    'u': [('U','U'), ('E','E'), ('EG','EG')],
    # *    'eps': [('E','E'), ('EG', 'EG')]
    # *}

    interpolate = kwargs.pop("interp_fnct", None)
    if interpolate:
        if interpolate == "dolfin.interpolate":
            interpolate = fe.interpolate
        elif interpolate == "interpolate_nonmatching_mesh_any":
            interpolate = interpolate_nonmatching_mesh_any
        else:
            interpolate = fe.interpolate
    else:
        interpolate = fe.interpolate

    reconstructed_fields = dict()

    for mecha_field in output_request:
        # * Prepare function spaces and assigner
        fspace = function_spaces[mecha_field]
        value_shape = fspace.ufl_element()._value_shape
        if len(value_shape) == 1:
            vector_dim = value_shape[0]
            mesh = fspace.mesh()
            element = fe.FiniteElement(
                fspace._ufl_element._family,
                mesh.ufl_cell(),
                fspace._ufl_element._degree,
            )
            element_family = element.family()
            constrain = fspace.dofmap().constrained_domain
            scalar_fspace = fe.FunctionSpace(
                mesh, element, constrained_domain=constrain
            )
            assigner = fe.FunctionAssigner(fspace, [scalar_fspace] * vector_dim)
            logger.debug(
                f"for reconstruction of {mecha_field}, vector_dim = {vector_dim}"
            )
        elif len(value_shape) == 0:
            logger.warning(
                "The value_shape attribute has not been found for the following function space. It is therefore assumed to be a scalar function space for the reconstruction : %s",
                fspace,
            )
            scalar_fspace = fspace
            vector_dim = 1
            element_family = fspace._ufl_element._family
            assigner = fe.FunctionAssigner(fspace, scalar_fspace)
        else:
            raise NotImplementedError(
                "Only vector fields are supported by the reconstruction function for now."
            )

        macro_kin_funct = dict()
        for key, field in macro_kinematic.items():
            macro_kin_funct[key] = list()
            for comp in field:
                if comp:
                    macro_kin_funct[key].append(interpolate(comp, scalar_fspace))
                else:
                    macro_kin_funct[key].append(0)

        # * Reconstruction proper
        contributions = [list() for i in range(vector_dim)]
        for macro_key, localization_key in localization_rules[mecha_field]:
            macro_f = macro_kin_funct[macro_key]
            loc_tens = localization_tensors[localization_key][mecha_field]
            for macro_comp, loc_tens_comps in zip(macro_f, loc_tens):
                if not macro_comp:
                    continue
                for i in range(vector_dim):
                    loc_comp = interpolate(loc_tens_comps[i], scalar_fspace)
                    contributions[i].append((macro_comp, loc_comp))

        # components = [sum(compnt_contrib) for compnt_contrib in contributions]
        components = [fe.Function(scalar_fspace) for i in range(vector_dim)]
        for i in range(vector_dim):
            vec = components[i].vector()
            values = vec.get_local()
            new_val = np.zeros_like(values)
            for macro_kin, loc in contributions[i]:
                loc_local_val = loc.vector().get_local()
                kin_local_val = macro_kin.vector().get_local()
                new_val += loc_local_val * kin_local_val
            values[:] = new_val
            vec.set_local(values)
            vec.apply("insert")
        # TODO : Regarder si le passage par np array faire perdre du temps, et auquel cas si l'on peut s'en passer.
        # TODO : avec axpy par exemple.
            # https://fenicsproject.org/docs/dolfin/2016.2.0/cpp/programmers-reference/la/GenericVector.html #noqa

        # * Components -> vector field
        field = fe.Function(fspace)
        if len(value_shape) == 0:
            components = components[0]
        assigner.assign(field, components)
        reconstructed_fields[mecha_field] = field
    return reconstructed_fields
