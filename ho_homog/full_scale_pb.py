# coding: utf8
"""
Created on 08/04/2019
@author: baptiste
"""

import logging
import dolfin as fe
import ho_homog
import numpy as np
from pathlib import Path


logger = logging.getLogger(__name__)

GEO_TOLERANCE = ho_homog.GEO_TOLERANCE
mat = ho_homog.materials

# * For mechanical fields reconstruction
MACRO_FIELDS_NAMES = ['U', 'E', 'EG', 'EGG']


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
                bc = [bc_['type'], bc_['constraint']]
                try:
                    bc.append(bc_['facet_function'])
                    bc.append(bc_['facet_idx'])
                except KeyError:
                    pass
            bc_type, *bc_data = bc
            if bc_type == 'Periodic':
                if self.per_bc is not None:
                    raise AttributeError("Only one periodic boundary condition can be prescribed.")
                self.per_bc = bc_data[0]
            elif bc_type == 'Dirichlet':
                if len(bc_data) == 2 or len(bc_data) == 3:
                    bc_data = tuple(bc_data)
                else:
                    raise AttributeError("Too much parameter for the definition of a Dirichlet boundary condition.")
                self.Dirichlet_bc.append(bc_data)

        self.measures = {self.part.dim: fe.dx, self.part.dim-1: fe.ds}

        # * Function spaces
        try:
            self.elmt_family = family = element['family']
            self.elmt_degree = degree = element['degree']
        except TypeError:  # Which means that element is not a dictionnary
            self.element_family, self.element_degree = element
            family, degree = element
        cell = self.part.mesh.ufl_cell()
        Voigt_strain_dim = int(self.part.dim*(self.part.dim+1)/2)
        self.scalar_fspace = fe.FunctionSpace(
            self.part.mesh,
            fe.FiniteElement(family, cell, degree),
            constrained_domain=self.per_bc)
        self.displ_fspace = fe.FunctionSpace(
            self.part.mesh,
            fe.VectorElement(family, cell, degree, dim=self.part.dim),
            constrained_domain=self.per_bc)
        self.strain_fspace = fe.FunctionSpace(
            self.part.mesh,
            fe.VectorElement(family, cell, degree, dim=Voigt_strain_dim),
            constrained_domain=self.per_bc)

        self.v = fe.TestFunction(self.displ_fspace)
        self.u = fe.TrialFunction(self.displ_fspace)
        self.a = fe.inner(
                    mat.sigma(self.part.elasticity_tensor, mat.epsilon(self.u)),
                    mat.epsilon(self.v)
                ) * self.measures[self.part.dim]
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

    def set_solver(self, solver_type='LU', solver_method='mumps', preconditioner='default'):
        """
        Choose the type of the solver and its method.

        An up-to-date list of the available solvers and preconditioners
        can be obtained with dolfin.list_linear_solver_methods() and
        dolfin.list_krylov_solver_preconditioners().
        """

        if solver_type == 'Krylov':
            self.solver = fe.KrylovSolver(self.K, solver_method)
        elif solver_type == 'LU':
            self.solver = fe.LUSolver(self.K, solver_method)
        else:
            raise TypeError
        if preconditioner != 'default':
            self.solver.parameters.preconditioner = preconditioner
        self._solver = {'type': solver_type, 'method': solver_method,
                        'preconditioner': preconditioner}
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
                        raise ValueError("Only one mesh function for each topology dimension can be use for the definition of loads.")
                except KeyError:
                    self.load_subdomains[topo_dim] = mesh_fctn

        # * Define load integrals
        labels = {self.part.dim: 'dx', self.part.dim-1: 'ds'}
        for topo_dim, partition in self.load_subdomains.items():
            self.measures[topo_dim] = fe.Measure(
                labels[topo_dim],
                domain=self.part.mesh,
                subdomain_data=partition)

        self.load_integrals = dict()
        for topo_dim, loads in self.loads_data.items():
            self.load_integrals[topo_dim] = list()
            dy = self.measures[topo_dim]
            for load in loads:
                if len(load) == 1:
                    contrib = fe.dot(load[0], self.v) * dy
                elif len(load) == 2:
                    contrib = fe.dot(load[1]*load[0], self.v) * dy
                elif len(load) == 3:
                    contrib = fe.dot(load[0], self.v) * dy(load[2])
                else:
                    raise AttributeError("Too much parameter for the definition of a load. Expecting 2, 3 or 4 parameters for each load.")
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
            L = 0.

        logger.info('Assembling system...')
        K, res = fe.assemble_system(self.a, L, self.Dirichlet_bc)
        logger.info('Assembling system : done')
        self.u_sol = fe.Function(self.displ_fspace)
        logger.info('Solving system...')
        self.solver.solve(K, self.u_sol.vector(), res)
        logger.info('Solving system : done')
        logger.info('Computing strain solution...')
        self.eps_sol = fe.project(self.part.epsilon(self.u_sol),
                                  self.strain_fspace,
                                  solver_type=self._solver['method'])
        logger.info('Saving results...')
        if results_file is not None:
            try:
                if results_file.suffix != '.xdmf':
                    results_file = results_file.with_suffix('.xdmf')
            except AttributeError:
                results_file = Path(results_file).with_suffix('.xdmf')
            with fe.XDMFFile(results_file.as_posix()) as file_out:
                file_out.parameters["flush_output"] = False
                file_out.parameters["functions_share_mesh"] = True
                self.u_sol.rename('displacement', 'displacement solution, full scale problem')
                self.eps_sol.rename('strain', 'strain solution, full scale problem')
                file_out.write(self.u_sol, 0.)
                file_out.write(self.eps_sol, 0.)

        return self.u_sol


class PeriodicDomain(fe.SubDomain):
    """Representation of periodicity boundary conditions. For 2D only"""
    # ? Source : https://comet-fenics.readthedocs.io/en/latest/demo/periodic_homog_elas/periodic_homog_elas.html

    def __init__(self, per_vectors, master_tests, slave_tests, dim=2, tolerance=GEO_TOLERANCE):
        fe.SubDomain.__init__(self, tolerance)
        self.tol = tolerance
        self.dim = dim
        self.per_vectors = per_vectors
        self.master_tests = master_tests
        self.slave_tests = slave_tests
        self.infinity = list()
        for i in range(self.dim):
            val = 9999 * sum([vect[i] for vect in self.per_vectors])
            self.infinity.append(val)

    def inside(self, x, on_boundary):
        """ Detect if point x is on a master part of the boundary."""
        if not on_boundary:
            return False
        if any(master(x) for master in self.master_tests):
            if not any(slave(x) for slave in self.slave_tests):
                return True
        else:
            return False

    def map(self, x, y):
        """ Link a point 'x' on a slave part of the boundary to the related point 'y' which belong to a master region."""
        slave_flag = False
        for slave, translat in zip(self.slave_tests, self.per_vectors):
            if slave(x):
                for i in range(self.dim):
                    y[i] = x[i] - translat[i]
                slave_flag = True
        if not slave_flag:
            for i in range(self.dim):
                y[i] = self.infinity[i]

    @staticmethod
    def pbc_dual_base(part_vectors, per_choice: str, dim=2, tolerance=GEO_TOLERANCE):
        """Create periodic boundary only array that indicate the dimensions of the part. Appropriate for parallelepipedic domain.

        Parameters
        ----------
        part_vectors : np.array
            shape 2×2. Dimensions of the domain.
            Some of them will be used as periodicity vectors.
        per_choice : str
            Can contain X, Y (in the future : Z)
        dim : int, optional
            Dimension of the modeling space. (the default is 2)
        tolerance : float, optional
            geometrical tolerance for membership tests.

        Returns
        -------
        PeriodicDomain
        """
        dual_vect = np.linalg.inv(part_vectors).T
        basis, dualbasis = list(), list()
        for i in range(np.size(part_vectors, 1)):
            basis.append(fe.as_vector(part_vectors[:, i]))
            dualbasis.append(fe.as_vector(dual_vect[:, i]))
        master_tests, slave_tests, per_vectors = list(), list(), list()
        if 'x' in per_choice.lower():
            def left(x):
                return fe.near(x.dot(dualbasis[0]), 0., tolerance)
                # dot product return a <'ufl.constantvalue.FloatValue'>

            def right(x):
                return fe.near((x - basis[0]).dot(dualbasis[0]), 0., tolerance)
            master_tests.append(left)
            slave_tests.append(right)
            per_vectors.append(basis[0])
        if 'y' in per_choice.lower():
            def bottom(x):
                return fe.near(x.dot(dualbasis[1]), 0., tolerance)

            def top(x):
                return fe.near((x - basis[1]).dot(dualbasis[1]), 0., tolerance)
            master_tests.append(bottom)
            slave_tests.append(top)
            per_vectors.append(basis[1])
        return PeriodicDomain(per_vectors, master_tests, slave_tests,
                              dim, tolerance)

    @staticmethod
    def pbc_facet_function(part_vectors, mesh, facet_function, per_choice: dict,
                           dim=2, tolerance=GEO_TOLERANCE):
        """[summary]

        Parameters
        ----------
        part_vectors : np.array
        mesh : Mesh
        facet_function : MeshFunction
        per_choice : dict
            key can be : 'X', 'Y'
            values : tuple (value of facetfunction for master, value for slave)
            Ex : {'X' : (3,5)}
        tolerance : float, optional

        Returns
        -------
        PeriodicDomain
        """

        # ! Not tested yet
        basis = list()
        for i in range(np.size(part_vectors, 1)):
            basis.append(fe.as_vector(part_vectors[:, i]))
        per_values = [val for couple in per_choice for val in couple]
        coordinates = dict()
        mesh.init(1, 0)
        for val in per_values:
            points_for_val = list()
            facet_idces = facet_function.where_equal(val)
            for i in facet_idces:
                vertices_idces = fe.Facet(mesh, i).entities(0)
                for j in vertices_idces:
                    coord = fe.Vertex(mesh, j).point().array()
                    points_for_val.append(coord)
            coordinates[val] = points_for_val
        master_tests, slave_tests, per_vectors = list(), list(), list()
        for key, (master_idx, slave_idx) in per_choice.items():
            def master_test(x):
                return any(np.allclose(x, pt, atol=tolerance) for pt in coordinates[master_idx])

            def slave_test(x):
                return any(np.allclose(x, pt, atol=tolerance) for pt in coordinates[slave_idx])
            master_tests.append(master_test)
            slave_tests.append(slave_test)
            if key.lower() == 'x':
                per_vectors.append(basis[0])
            elif key.lower() == 'y':
                per_vectors.append(basis[1])

        return PeriodicDomain(per_vectors, master_tests, slave_tests,
                              dim, tolerance)


def reconstruction(
    localization_tensors: dict, macro_kinematic: dict, function_spaces: dict,
    localization_rules: dict = {}, trunc_order: int = 0,
    output_request=('u', 'eps'), proj_solver=None):
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
        keys : 'u', 'eps' or 'sigma'
        values : FEniCS function space
    localization_rules : dict, optional
        Indicates the rules that have to be followed for the construction of the fluctuations.
        (the default is {}, which means that the trunc_order argument will be used)
    trunc_order : int, optional
        Order of truncation for the reconstruction of the displacement field
        following the higher order homogenization scheme defined in ???.
    output_request : tuple of strings, optional
        Which fields have to be calculated.
        This can contain : 'u', eps' and 'sigma'
        (the default is ('u', 'eps'), displacement and strain fields will be reconstructed)
    proj_solver : string
        impose the use of a desired solver for the projections.

    Return
    ------
    Dictionnary
        Mechanical fields with microscopic fluctuations. Keys are "eps", "sigma" and "u" respectively for the strain, stress and displacement fields.
    """
    # ? Changer les inputs ?
        # ? Remplacer le dictionnaire de functionspace et output_request par un seul argument : list de tuples qui contiennent nom + function_space ?
        # ? Comme ça on peut aussi imaginer reconstruire le même champs dans différents espaces ?

    # ! La construction de champs de localisation périodiques doit être faite en dehors de cette fonction.

    # TODO : récupérer topo_dim à partir des tenseurs de localisation, ou mieux, à partir des espaces fonctionnels
    # TODO : choisir comment on fixe la len des listes correspondantes aux composantes de u et de epsilon.

    # TODO : permettre la translation du RVE par rapport à la structure macro autre part
    # TODO :  translation_microstructure: np.array, optional
    # TODO :         Vector, 1D array (shape (2,) or (3,)), position origin used for the description of the RVE with regards to the macroscopic origin.

    solver_param = {}
    if proj_solver:
        solver_param = {"solver_type": proj_solver}

    # Au choix, utilisation de trunc_order ou localization_rules dans les kargs
    if localization_rules:
        pass
    elif trunc_order:
        localization_rules = {
            'u': [(MACRO_FIELDS_NAMES[i], MACRO_FIELDS_NAMES[i]) for i in range(trunc_order+1)],
            'eps': [(MACRO_FIELDS_NAMES[i], MACRO_FIELDS_NAMES[i]) for i in range(1, trunc_order+1)]
        }
    # * Ex. for truncation order = 2:
    # * localization_rules = {
    # *    'u': [('U','U'), ('E','E'), ('EG','EG')],
    # *    'eps': [('E','E'), ('EG', 'EG')]
    # *}

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
                fspace._ufl_element._degree
                )
            constrain = fspace.dofmap().constrained_domain
            scalar_fspace = fe.FunctionSpace(mesh, element, constrained_domain=constrain)
            assigner = fe.FunctionAssigner(fspace, [scalar_fspace]*vector_dim)
            logger.debug(f"for reconstruction of {mecha_field}, vector_dim = {vector_dim}")
        elif len(value_shape) == 0:
            logger.warning("The value_shape attribute has not been found for the following function space. It is therefore assumed to be a scalar function space for the reconstruction : %s", fspace)
            scalar_fspace = fspace
            vector_dim = 1
            assigner = fe.FunctionAssigner(fspace, scalar_fspace)
        else:
            raise NotImplementedError("Only vector fields are supported by the reconstruction function for now.")

        # * Reconstruction proper
        contributions = [list() for i in range(vector_dim)]
        for macro_key, localization_key in localization_rules[mecha_field]:
            macro_f = macro_kinematic[macro_key]
            loc_tens = localization_tensors[localization_key][mecha_field]
            for macro_comp, loc_tens_comps in zip(macro_f, loc_tens):
                if not macro_comp:
                    continue
                for i in range(vector_dim):
                    contributions[i].append(macro_comp * loc_tens_comps[i])
        components = [sum(compnt_contrib) for compnt_contrib in contributions]

        # * Components -> vector field
        field = fe.Function(fspace)
        components_proj = list()
        for scl_field in components:
            components_proj.append(
                fe.project(scl_field, scalar_fspace, **solver_param))
        if len(value_shape) == 0:
            components_proj = components_proj[0]
        assigner.assign(field, components_proj)
        reconstructed_fields[mecha_field] = field
    return reconstructed_fields
