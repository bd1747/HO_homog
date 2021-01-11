# coding: utf8
"""
Created on 08/04/2019
@author: baptiste
"""

import logging
import dolfin as fe
from ho_homog import full_scale_pb as fsp
from pathlib import Path
from pytest import approx

fe.set_log_level(30)


def test_reconstruction_vectors():
    logger = logging.getLogger("test_reconstruction")
    nb_x = nb_y = 30
    L_x = 2 * fe.pi
    L_y = 2 * fe.pi
    mesh = fe.RectangleMesh(fe.Point(0, -L_y / 2), fe.Point(L_x, L_y / 2), nb_x, nb_y)
    elem_type = "CG"
    degree = 2
    displ_fspace = fe.VectorFunctionSpace(mesh, elem_type, degree)
    strain_fspace = fe.FunctionSpace(
        mesh, fe.VectorElement(elem_type, mesh.ufl_cell(), degree, dim=3)
    )
    scalar_fspace = fe.FunctionSpace(
        mesh, fe.FiniteElement(elem_type, mesh.ufl_cell(), degree)
    )

    macro_fields = {
        "U": [fe.Expression("0.5*x[0]", degree=1), 0],
        "E": [fe.Expression("0.5", degree=0), 0, 0],
    }
    localization_tensors = {
        "U": {
            "u": [
                fe.interpolate(fe.Constant((1.0, 0.0)), displ_fspace).split(),
                fe.interpolate(fe.Constant((0.0, 1.0)), displ_fspace).split(),
            ]
        },
        "E": {
            "u": [
                [
                    fe.Expression("cos(x[0])", degree=2),
                    fe.Expression("sin(x[1])", degree=2),
                ],
                [],
                [],
            ],
            "eps": [
                [
                    fe.Expression("1-sin(x[0])", degree=2),
                    fe.Expression("cos(x[1])", degree=2),
                    fe.Constant(value=0),
                ],
                [],
                [],
            ],
        },
    }
    function_spaces = {"u": displ_fspace, "eps": strain_fspace}
    exact_sol = {
        "u": fe.project(
            fe.Expression(("0.5*x[0] + 0.5*cos(x[0])", "0.5*sin(x[1])"), degree=2),
            displ_fspace,
        ),
        "eps": fe.project(
            fe.Expression((" 0.5 - 0.5*sin(x[0])", "0.5*cos(x[1])", "0"), degree=2),
            strain_fspace,
        ),
    }
    reconstr_sol = fsp.reconstruction(
        localization_tensors, macro_fields, function_spaces, trunc_order=1
    )
    results_file_path = Path(__file__).with_name("test_reconstruction.xdmf")
    with fe.XDMFFile(str(results_file_path)) as results_file:
        data = [
            (exact_sol["u"], "disp_exact", "exact displacement"),
            (exact_sol["eps"], "strain_exact", "exact strain"),
            (reconstr_sol["u"], "disp_reconstruction", "displacement reconstruction"),
            (reconstr_sol["eps"], "strain_reconstruction", "strain reconstruction"),
            (
                fe.project(macro_fields["U"][0], scalar_fspace),
                "disp_macro",
                "disp_macro",
            ),
            (
                fe.project(macro_fields["E"][0], scalar_fspace),
                "strain_macro",
                "strain_macro",
            ),
        ]
        results_file.parameters["flush_output"] = False
        results_file.parameters["functions_share_mesh"] = True
        for field, name, descrpt in data:
            field.rename(name, descrpt)
            results_file.write(field, 0.0)
    for f_name in reconstr_sol.keys():
        dim = exact_sol[f_name].ufl_shape[0]
        exact_norm = fe.norm(exact_sol[f_name], "L2", mesh)
        difference = fe.errornorm(
            exact_sol[f_name], reconstr_sol[f_name], "L2", mesh=mesh
        )
        error = difference / exact_norm
        logger.debug(f"Relative error for {f_name} = {error}")
        # * ref: Relative errors for u = 3.504e-15; for eps = 2.844e-16
        assert error == approx(0.0, abs=1e-13)


def test_reconstruction_with_constraint():
    logger = logging.getLogger("test_reconstruction")
    nb_x = nb_y = 20
    L_x = 2
    L_y = 2
    mesh = fe.RectangleMesh(
        fe.Point(-L_x / 2, -L_y / 2), fe.Point(L_x / 2, L_y / 2), nb_x, nb_y
    )

    class PeriodicDomain(fe.SubDomain):
        # ? Source : https://comet-fenics.readthedocs.io/en/latest/demo/periodic_homog_elas/periodic_homog_elas.html
        def __init__(self, tolerance=fe.DOLFIN_EPS):
            """ vertices stores the coordinates of the 4 unit cell corners"""
            fe.SubDomain.__init__(self, tolerance)
            self.tol = tolerance

        def inside(self, x, on_boundary):
            Bottom = fe.near(x[1], -L_y / 2, self.tol)
            return Bottom and on_boundary

        def map(self, x, y):
            SlaveT = fe.near(x[1], L_y / 2.0, self.tol)
            if SlaveT:
                for i in range(len(x[:])):
                    y[i] = (x[i] - L_y) if i == 1 else x[i]
            else:
                for i in range(len(x[:])):
                    y[i] = 1000.0 * (L_x + L_y)

    pbc = PeriodicDomain()
    elem_type = "CG"
    degree = 2
    scalar_fspace = fe.FunctionSpace(
        mesh,
        fe.FiniteElement(elem_type, mesh.ufl_cell(), degree),
        constrained_domain=pbc,
    )

    macro_fields = {
        "U": [fe.Expression("0.5*x[0]+x[1]", degree=1)],
        "E": [fe.Expression("0.5", degree=0), fe.Expression("1", degree=0)],
    }
    localization_tensors = {
        "U": {"u": [[fe.interpolate(fe.Constant(1.0), scalar_fspace)]]},
        "E": {
            "u": [[fe.Expression("x[1]", degree=2)], [fe.Expression("x[0]", degree=2)]]
        },
    }
    function_spaces = {"u": scalar_fspace}
    localization_rules = {"u": [("U", "U"), ("E", "E")]}

    exact_sol = fe.project(
        fe.Expression("0.5*x[0] + x[1] + 0.5*x[1] + 1.*x[0]", degree=2), scalar_fspace
    )

    reconstr_sol = fsp.reconstruction(
        localization_tensors,
        macro_fields,
        function_spaces,
        localization_rules=localization_rules,
        output_request=("u",),
    )
    reconstr_sol = reconstr_sol["u"]
    results_file_path = Path(__file__).with_name("test_reconstruction_constraint.xdmf")
    with fe.XDMFFile(str(results_file_path)) as results_file:
        data = [
            (exact_sol, "sol_exact", "exact solution field"),
            (reconstr_sol, "sol_reconstructed", "reconstructed solution field"),
        ]
        results_file.parameters["flush_output"] = False
        results_file.parameters["functions_share_mesh"] = True
        for field, name, descrpt in data:
            field.rename(name, descrpt)
            results_file.write(field, 0.0)
    exact_norm = fe.norm(exact_sol, "L2", mesh)
    difference = fe.errornorm(exact_sol, reconstr_sol, "L2", mesh=mesh)
    error = difference / exact_norm
    logger.debug(f"Relative error = {error}")  # * ref: 3.2625e-15
    assert error == approx(0.0, abs=1e-14)


def test_select_solver():
    """The solver Mumps is selected."""
    logger = logging.getLogger("test_reconstruction")
    nb_x = nb_y = 20
    L_x = 2
    L_y = 2
    mesh = fe.RectangleMesh(
        fe.Point(-L_x / 2, -L_y / 2), fe.Point(L_x / 2, L_y / 2), nb_x, nb_y
    )

    class PeriodicDomain(fe.SubDomain):
        def __init__(self, tolerance=fe.DOLFIN_EPS):
            """ vertices stores the coordinates of the 4 unit cell corners"""
            fe.SubDomain.__init__(self, tolerance)
            self.tol = tolerance

        def inside(self, x, on_boundary):
            Bottom = fe.near(x[1], -L_y / 2, self.tol)
            return Bottom and on_boundary

        def map(self, x, y):
            SlaveT = fe.near(x[1], L_y / 2.0, self.tol)
            if SlaveT:
                for i in range(len(x[:])):
                    y[i] = (x[i] - L_y) if i == 1 else x[i]
            else:
                for i in range(len(x[:])):
                    y[i] = 1000.0 * (L_x + L_y)

    pbc = PeriodicDomain()
    scalar_fspace = fe.FunctionSpace(
        mesh, fe.FiniteElement("CG", mesh.ufl_cell(), 2), constrained_domain=pbc
    )
    macro_fields = {
        "U": [fe.Expression("0.5*x[0]+x[1]", degree=1)],
        "E": [fe.Expression("0.5", degree=0), fe.Expression("1", degree=0)],
    }
    localization_tensors = {
        "U": {"u": [[fe.interpolate(fe.Constant(1.0), scalar_fspace)]]},
        "E": {
            "u": [[fe.Expression("x[1]", degree=2)], [fe.Expression("x[0]", degree=2)]]
        },
    }
    function_spaces = {"u": scalar_fspace}
    localization_rules = {"u": [("U", "U"), ("E", "E")]}
    exact_sol = fe.project(
        fe.Expression("0.5*x[0] + x[1] + 0.5*x[1] + 1.*x[0]", degree=2), scalar_fspace
    )

    reconstr_sol = fsp.reconstruction(
        localization_tensors,
        macro_fields,
        function_spaces,
        localization_rules=localization_rules,
        output_request=("u",),
        proj_solver="mumps",
    )
    reconstr_sol = reconstr_sol["u"]
    exact_norm = fe.norm(exact_sol, "L2", mesh)
    difference = fe.errornorm(exact_sol, reconstr_sol, "L2", mesh=mesh)
    error = difference / exact_norm
    logger.debug(f"Relative error = {error}")  # * ref: 3.2625e-15
    assert error == approx(0.0, abs=1e-14)


if __name__ == "__main__":
    logger_root = logging.getLogger()
    logger_root.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s :: %(levelname)s :: %(message)s", "%H:%M"
    )
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger_root.addHandler(stream_handler)

    test_reconstruction_vectors()
    test_reconstruction_with_constraint()
