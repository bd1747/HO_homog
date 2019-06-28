# coding: utf8
"""
Created on 08/04/2019
@author: baptiste
"""

import logging
import dolfin as fe
from ho_homog import GEO_TOLERANCE
import numpy as np

logger = logging.getLogger(__name__)


class PeriodicDomain(fe.SubDomain):
    """Representation of periodicity boundary conditions. For 2D only"""

    # ? Source : https://comet-fenics.readthedocs.io/en/latest/demo/periodic_homog_elas/periodic_homog_elas.html #noqa

    def __init__(
        self, per_vectors, master_tests, slave_tests, dim=2, tol=GEO_TOLERANCE
    ):
        fe.SubDomain.__init__(self, tol)
        self.tol = tol
        self.dim = dim
        self.per_vectors = tuple([np.asarray(v, "float64") for v in per_vectors])
        self.infinity = sum([9999 * v for v in self.per_vectors])
        self.master_tests = master_tests
        self.slave_tests = slave_tests

    def inside(self, x, on_boundary):
        """ Detect if point x is on a master part of the boundary."""
        if not on_boundary:
            return False
        if any(master(x) for master in self.master_tests):
            if not any(slave(x) for slave in self.slave_tests):
                return True
        return False

    def map(self, x, y):
        """
        Link a point 'x' on a slave part of the boundary
        to the related point 'y' which belong to a master region.
        """
        translation = np.zeros(self.dim)
        for s_test, v in zip(self.slave_tests, self.per_vectors):
            if s_test(x):
                translation -= v
        if translation.any():
            for i in range(self.dim):
                y[i] = x[i] + translation[i]
        else:
            for i in range(self.dim):
                y[i] = self.infinity[i]

    @staticmethod
    def pbc_dual_base(part_vectors, per_choice: str, dim=2, tol=GEO_TOLERANCE):
        """Create periodic boundary only from an array
        that indicate the dimensions of the part.
        Appropriate for parallelepipedic domain.

        Parameters
        ----------
        part_vectors : np.array
            shape 2×2. Dimensions of the domain.
            Some of them will be used as periodicity vectors.
        per_choice : str
            Can contain X, Y (in the future : Z)
        dim : int, optional
            Dimension of the modeling space. (the default is 2)
        tol : float, optional
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
        if "x" in per_choice.lower():

            def left(x):
                return fe.near(x.dot(dualbasis[0]), 0.0, tol)
                # dot product return a <'ufl.constantvalue.FloatValue'>

            def right(x):
                return fe.near((x - basis[0]).dot(dualbasis[0]), 0.0, tol)

            master_tests.append(left)
            slave_tests.append(right)
            per_vectors.append(basis[0])
        if "y" in per_choice.lower():

            def bottom(x):
                return fe.near(x.dot(dualbasis[1]), 0.0, tol)

            def top(x):
                return fe.near((x - basis[1]).dot(dualbasis[1]), 0.0, tol)

            master_tests.append(bottom)
            slave_tests.append(top)
            per_vectors.append(basis[1])
        return PeriodicDomain(per_vectors, master_tests, slave_tests, dim, tol)

    @staticmethod
    def pbc_facet_function(
        part_vectors, mesh, facet_function, per_choice: dict, dim=2, tol=GEO_TOLERANCE
    ):
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
        tol : float, optional

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
                return any(
                    np.allclose(x, pt, atol=tol) for pt in coordinates[master_idx]
                )

            def slave_test(x):
                return any(
                    np.allclose(x, pt, atol=tol) for pt in coordinates[slave_idx]
                )

            master_tests.append(master_test)
            slave_tests.append(slave_test)
            if key.lower() == "x":
                per_vectors.append(basis[0])
            elif key.lower() == "y":
                per_vectors.append(basis[1])

        return PeriodicDomain(per_vectors, master_tests, slave_tests, dim, tol)


# * Periodic localization fields
per_scalar_fnct_cpp_code = """
    #include <pybind11/pybind11.h>
    #include <pybind11/eigen.h>
    namespace py = pybind11;

    #include <dolfin/function/Expression.h>
    #include <dolfin/function/Function.h>

    class PeriodicExpr : public dolfin::Expression
    {
    public:

        PeriodicExpr() : dolfin::Expression() {}

        void eval(
            Eigen::Ref<Eigen::VectorXd> values,
            Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell
        ) const override
        {
        Eigen::Vector2d coord_equi;
        coord_equi[0] = x[0] -per_x*floor(x[0]/per_x);
        coord_equi[1] = x[1] -per_y*floor(x[1]/per_y);
        f->eval(values, coord_equi);
        }

    std::shared_ptr<dolfin::Function> f;
    double per_x;
    double per_y;
    };

    PYBIND11_MODULE(SIGNATURE, m)
    {
    py::class_<PeriodicExpr, std::shared_ptr<PeriodicExpr>, dolfin::Expression>
        (m, "PeriodicExpr", py::dynamic_attr())
        .def(py::init<>())
        .def_readwrite("f", &PeriodicExpr::f)
        .def_readwrite("per_x", &PeriodicExpr::per_x)
        .def_readwrite("per_y", &PeriodicExpr::per_y);
    }
    """
periodic_cppcode = fe.compile_cpp_code(per_scalar_fnct_cpp_code)


class PeriodicExpr(fe.UserExpression):
    """Periodic extension of a (periodic) function. Represented by an expression"""

    def __init__(self, base_function, per_vectors, topo_dim=2, **kwargs):
        """Create a periodic extension of a function which is defined only on a cell.

        For now, only the case of periodic vectors aligned with the global basis
        in 2D is supported.

        Parameters
        ----------
        base_function : dolfin Function
            Function that will be extended by periodicity.
            Must be at least defined on a cell delimited by the 4 points :
            O, vp1, vp2, vp1+vp2 where vp1 and vp2 are the periodicity vectors.
        per_vectors : ndarray
            periodicity vectors in column
        topo_dim : int, optional
            [description], by default 2
        """
        self.base_function = base_function
        self.per_vectors = np.asarray(per_vectors)
        assert self.per_vectors.shape == (2, 2), "Invalid shape. Only 2D supported"
        diag_test = np.allclose(self.per_vectors[[0, 1], [1, 0]], [0.0, 0.0], atol=1e-12)
        assert diag_test, "Periodicity vectors must be align with global basis."
        # TODO : généralisation à des vecteurs de périodicité non alignés avec le repère.

        super().__init__(degree=kwargs["degree"])

        per_cpp_expr = periodic_cppcode.PeriodicExpr()
        per_cpp_expr.per_x = self.per_vectors[0, 0]
        per_cpp_expr.per_y = self.per_vectors[1, 1]
        base_function.set_allow_extrapolation(True)
        per_cpp_expr.f = base_function.cpp_object()
        self._cpp_object = per_cpp_expr

    def value_shape(self):
        return ()
