# coding: utf8
"""
Created on 03/05/2019
@author: baptiste
"""

import dolfin as fe
import ho_homog.toolbox_FEniCS as tb
from pytest import approx

fe.set_log_level(30)


def test_function_errornorm():
    mesh = fe.UnitSquareMesh(20, 20)
    V = fe.FunctionSpace(mesh, "Lagrange", 2)
    W = fe.FunctionSpace(mesh, "Lagrange", 2)

    v = fe.Function(V)
    w = fe.Function(W)

    expr_v = fe.Expression('x[0]*x[0]+2*x[1]', degree=2)
    expr_w = fe.Expression('x[0]*x[0]+2*x[1]+0.01', degree=5)
    v = fe.interpolate(expr_v, V)
    w = fe.project(expr_w, W)
    err = tb.function_errornorm(v, w, enable_diff_fspace=True)
    assert err == approx(0.01)
    w_bis = fe.project(expr_w, V)
    err_bis = tb.function_errornorm(v, w_bis, enable_diff_fspace=False)