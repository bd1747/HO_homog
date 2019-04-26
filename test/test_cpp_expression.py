# coding: utf-8
"""
Created on 24/04/2019
@author: baptiste

Test : scalar-valued function defined with c++ snippet.

Check that we can find the C_{1111}^{hom} theoretical value for laminate.

"""
import dolfin as fe
import numpy as np
import matplotlib.pyplot as plt
from pytest import approx


def test_cpp_coeff_material(plots=False):
    """Test : scalar-valued function defined with c++ snippet.

    Check that we can find the C_{1111}^{hom} theoretical value for laminate.
    """
    mesh = fe.UnitSquareMesh(20, 20)

    tol = 1e-14
    stiff_width = 0.5

    class StiffDomain(fe.SubDomain):
        def inside(self, x, on_boundary):
            left_limit = x[1] >= 0.5 - stiff_width/2 - tol
            right_limit = x[1] <= 0.5 + stiff_width/2 + tol
            return left_limit and right_limit
    material_subdo = fe.MeshFunction('size_t', mesh, dim=2)
    material_subdo.set_all(0)
    stiff_domain = StiffDomain()
    stiff_domain.mark(material_subdo, 1)

    cppcode = """
    #include <pybind11/pybind11.h>
    #include <pybind11/eigen.h>
    namespace py = pybind11;

    #include <dolfin/function/Expression.h>
    #include <dolfin/mesh/MeshFunction.h>

    class MatCoeff : public dolfin::Expression
    {
    public:

        MatCoeff() : dolfin::Expression() {}

        void eval(Eigen::Ref<Eigen::VectorXd> values,
                  Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell
                 ) const override
        {
            if ((*materials)[cell.index] == 0)
            values[0] = val_0;
            else
            values[0] = val_1;
        }

    std::shared_ptr<dolfin::MeshFunction<std::size_t>> materials;
    double val_0;
    double val_1;
    };

    PYBIND11_MODULE(SIGNATURE, m)
    {
    py::class_<MatCoeff, std::shared_ptr<MatCoeff>, dolfin::Expression>
        (m, "MatCoeff", py::dynamic_attr())
        .def(py::init<>())
        .def_readwrite("materials", &MatCoeff::materials)
        .def_readwrite("val_0", &MatCoeff::val_0)
        .def_readwrite("val_1", &MatCoeff::val_1);
    }
    """

    E_soft = 1.0
    E_stiff = 10.0

    class UserYoungModulus(fe.UserExpression):
        def value_shape(self):
            return ()
    E = UserYoungModulus(degree=0)
    E_cppcode = fe.compile_cpp_code(cppcode).MatCoeff()
    E_cppcode.val_0 = E_soft
    E_cppcode.val_1 = E_stiff
    E_cppcode.materials = material_subdo
    E._cpp_object = E_cppcode

    # exterior facets MeshFunction
    class Top(fe.SubDomain):
        def inside(self, x, on_boundary):
            return fe.near(x[1], 1) and on_boundary

    class Left(fe.SubDomain):
        def inside(self, x, on_boundary):
            return fe.near(x[0], 0) and on_boundary

    class Bottom(fe.SubDomain):
        def inside(self, x, on_boundary):
            return fe.near(x[1], 0) and on_boundary

    class Right(fe.SubDomain):
        def inside(self, x, on_boundary):
            return fe.near(x[0], 1) and on_boundary
    facets = fe.MeshFunction("size_t", mesh, 1)
    facets.set_all(0)
    Top().mark(facets, 90)
    Left().mark(facets, 1)
    Bottom().mark(facets, 10)
    Right().mark(facets, 9)

    # Define variational problem
    V = fe.VectorFunctionSpace(mesh, "Lagrange", 2)
    X = fe.FunctionSpace(mesh, "Lagrange", 2)
    X_discontinuous = fe.FunctionSpace(mesh, "DG", 0)

    u = fe.TrialFunction(V)
    v = fe.TestFunction(V)

    nu = 0.3
    mu = 1/(2*(1+nu)) * E
    lmbda = nu/(1+nu)/(1-2*nu) * E

    def eps(v):
        return fe.sym(fe.grad(v))

    def sigma(v):
        return lmbda*fe.tr(eps(v))*fe.Identity(2) + 2.0*mu*eps(v)

    F = fe.inner(sigma(u), eps(v))*fe.dx
    a, L = fe.lhs(F), fe.rhs(F)

    bcs = [
        fe.DirichletBC(V.sub(1), fe.Constant(0.), facets, 90),  # Top
        fe.DirichletBC(V.sub(1), fe.Constant(0.), facets, 10),  # Bottom
        fe.DirichletBC(V.sub(0), fe.Constant(0.), facets, 1),  # Left
        fe.DirichletBC(V.sub(0), fe.Constant(1.), facets, 9),  # Right
    ]

    u = fe.Function(V, name="Displacement")
    fe.solve(a == L, u, bcs)
    strain_energy = 0.5 * fe.assemble(fe.inner(eps(u), sigma(u)) * fe.dx)

    if plots:
        data = [
            (material_subdo, {'title': 'subdomain MeshFunction'})
            (fe.project(lmbda, X_discontinuous), {'title': 'lambda'}),
            (u, {'mode': 'displacement', 'title': 'displacement'}),
            (fe.project(fe.inner(eps(u), sigma(u)), X), {'title': 'energy'}),
            (fe.project(eps(u)[(0, 0)], X), {'title': 'E_11'}),
            (fe.project(eps(u)[(1, 1)], X), {'title': 'E_22'}),
            (fe.project(eps(u)[(0, 1)], X), {'title': 'sqrt(2)*E_12'})
            ]
        for var, kwargs in data:
            plt.figure()
            p = fe.plot(var, **kwargs)
            fe.plot(mesh, linewidth=0.7)
            plt.colorbar(p)

    E = np.array((1.0, 10.0))
    nu = 0.3
    mu = 1/(2*(1+nu))*E
    lmbda = nu/(1+nu)/(1-2*nu) * E
    width = np.array((1.-stiff_width, stiff_width))
    C111_theo = (
        2*np.mean(width*mu)
        + 2*np.mean(width*mu*lmbda/(lmbda+2*mu))
        + (1/(np.mean(width*(1/(lmbda+2*mu))))
            * (np.mean(width*lmbda/(lmbda+2*mu)))**2))

    assert strain_energy == approx(C111_theo, rel=1e-9)


def cpp_per_expression():
    """Create a periodic expression from a fuction defined on a smaller domain"""
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

        void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const override
        {
        //    Eigen::Vector2d coord_equi;
        //coord_equi[0] = x[0] -per_x*floor(x[0]/per_x);
        //coord_equi[1] = x[1] -per_y*floor(x[1]/per_y);
        //f->eval(values, coord_equi);
        //passer par un pointeur ? *f->eval(values, coord_equi);
        //dummy val:
        values[0] = per_x + per_y;
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
