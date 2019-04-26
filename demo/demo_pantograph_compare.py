# coding: utf8
"""
Created on 18/04/2018
@author: baptiste

Compare the higher-order homogenized solution with the full-scale solution.
Example : 2D, pantograph microstructure.
"""
import logging
from pathlib import Path

import dolfin as fe
import gmsh
import numpy as np
import sympy as sp

from ho_homog import (GEO_TOLERANCE, full_scale_pb, geometry, homog2d,
                      materials, mesh_generate_2D, part, pckg_logger)

logger = logging.getLogger('demo_full_compare')
logger_root = logging.getLogger()
logger.setLevel(logging.INFO)
logger_root.setLevel(logging.DEBUG)

for hdlr in pckg_logger.handlers[:]:
    if isinstance(hdlr, logging.StreamHandler):
        pckg_logger.removeHandler(hdlr)
pckg_logger.setLevel(logging.INFO)
geometry.bndry_logger.setLevel(logging.DEBUG)
formatter = logging.Formatter(
    '%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s', "%H:%M:%S")
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
stream_handler.setFormatter(formatter)
logger_root.addHandler(stream_handler)

gmsh_logger = logging.getLogger('gmsh')
gmsh_logger.setLevel(logging.INFO)

logging.getLogger('UFL').setLevel(logging.DEBUG)
logging.getLogger('FFC').setLevel(logging.DEBUG)
fe.set_log_level(20)


def process_gmsh_log(gmsh_log: list, detect_error=True):
    """Treatment of log messages gathered with gmsh.logger.get()"""
    err_msg, warn_msg = list(), list()
    for line in gmsh_log:
        if "error" in line.lower():
            err_msg.append(line)
        if "warning" in line.lower():
            warn_msg.append(line)
    gmsh_logger.info("**********")
    gmsh_logger.info(f"{len(gmsh_log)} logging messages got from Gmsh : "
                     f"{len(err_msg)} errors, {len(warn_msg)} warnings.")
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


# * Step 1 : Modeling the geometry of the part and the RVE
geometry.init_geo_tools()
geometry.set_gmsh_option("Mesh.MaxNumThreads2D", 4)
geometry.set_gmsh_option("Mesh.MaxNumThreads1D", 4)
geometry.set_gmsh_option('Mesh.Algorithm', 5)
a = 1
b, k = a, a/3
r = 0.01
gmsh.logger.start()
part_geo = mesh_generate_2D.Gmsh2DRVE.pantograph(
                a, b, k, r,
                nb_cells=(20, 1),
                soft_mat=False, name='panto_part')
# ! Utiliser autre chose que Gmsh2DRVE
process_gmsh_log(gmsh.logger.get())
gmsh.logger.stop()
gmsh.logger.start()
rve_geo = mesh_generate_2D.Gmsh2DRVE.pantograph(
                a, b, k, r,
                nb_cells=(1, 1),
                soft_mat=False, name='panto_rve')
process_gmsh_log(gmsh.logger.get())
gmsh.logger.stop()

# * Step 2 : Generating the meshes
lc_ratio = 1/3.
d_min_max = (2*r*a, a)
lc_min_max = (lc_ratio*r*a, lc_ratio*a)
for geo_model in [part_geo, rve_geo]:
    geo_model.main_mesh_refinement(d_min_max, lc_min_max, False)
    gmsh.logger.start()
    geo_model.mesh_generate()
    process_gmsh_log(gmsh.logger.get())
    gmsh.logger.stop()

# * Step 2 : Defining the material properties
E, nu = 1., 0.3
E_nu_tuples = [(E, nu)]

material_dict_part = {
    part_geo.phy_surf[0].tag: materials.Material(E, nu, 'cp')}
material_dict_rve = {
    rve_geo.phy_surf[0].tag: materials.Material(E, nu, 'cp')}

# * Step 3 : Calculation of the full-scale solution
# * Step 3.1 : Create a part object
panto_part = part.Fenics2DRVE.file_2_FenicsPart(
                part_geo.mesh_abs_path, material_dict_part,
                global_dimensions=part_geo.gen_vect, subdomains_import=True,
                plots=False)

LX = panto_part.global_dimensions[0, 0]
LY = panto_part.global_dimensions[1, 1]


# * Step 3.2 : Definition of boundary conditions
def left_border(x, on_boundary):
    return on_boundary and x[0] <= GEO_TOLERANCE


pbc = full_scale_pb.PeriodicDomain.pbc_dual_base(
        panto_part.global_dimensions, 'Y')
boundary_conditions = [
    {'type': 'Periodic', 'constraint': pbc},
    ('Dirichlet', (0., 0.), left_border)
]

# * Step 3.3 : Definition of the load
load_width = Lx_cell = 4*a
load_x = LX/2
indicator_fctn = fe.Expression(
    'x[0] >= (loadx-loadw/2.0) && x[0] <= (loadx+loadw/2.0) ? 1 : 0',
    degree=1, loadx=load_x, loadw=load_width)
load_area = fe.assemble(indicator_fctn*fe.dx(panto_part.mesh))
load_magnitude = 1.
s_load = load_magnitude / load_area
s_load = fe.Constant((s_load, 0.))  # forme utilisée dans tuto FEniCS linear elasticity
load = (2, s_load, indicator_fctn)

# * Step 3.3 : Choosing FE characteristics
element = ('Lagrange', 2)

# * Step 3.4 : Gathering all data in a model
model = full_scale_pb.FullScaleModel(
    panto_part, [load], boundary_conditions, element)
model.set_solver('LU', 'mumps')

# * Step 3.5 : Solving the problem
results_file = Path('demo_results/pantograph_compare_full_scale.xdmf')
model.solve(results_file=results_file)

# * Step 4 : Calculating homogenized mechanical behavior
# * Step 4.1 : Creating the Python object that represents the RVE and
# * which is suitable for FEniCS
rve = part.Fenics2DRVE.gmsh_2_Fenics_2DRVE(rve_geo, material_dict_rve)

# * Step 4.2 : Initializing the homogemization model
hom_model = homog2d.Fenics2DHomogenization(rve)

# * Step 4.3 : Computing the homogenized consitutive tensors
*localization_tens_u_sig_eps, constitutive_tens = hom_model.homogenizationScheme('EGG')

# * Step 5 : Definition of the homogenized solution
x, F_sb, Q1_sb, D1_sb, A_sb, L_sb = sp.symbols('x[0], F, Q_1, D_1, A, L')

k = sp.sqrt(Q1_sb / D1_sb)
f0 = F_sb / (k**2 * D1_sb)

Am = -f0
Bm = f0 * (sp.sinh(k*A_sb) - sp.tanh(k*L_sb)*(sp.cosh(k*A_sb)-1))
Cm = -Bm / k
Ap = f0 * (sp.cosh(k*A_sb) - 1)
Bp = -f0 * sp.tanh(k*L_sb)*(sp.cosh(k*A_sb) - 1)
Cp = f0/k * (k*A_sb - sp.sinh(k*A_sb) + sp.tanh(k*L_sb)*(sp.cosh(k*A_sb)-1))

u_m_sb = [Am/k*sp.sinh(k*x) + Bm/k*sp.cosh(k*x) + f0*x + Cm]
u_p_sb = [Ap/k*sp.sinh(k*x) + Bp/k*sp.cosh(k*x) + Cp]
for i in range(1, 4):
    u_m_sb.append(u_m_sb[0].diff(x, i))
    u_p_sb.append(u_p_sb[0].diff(x, i))

Q_1_hom = constitutive_tens['E']['E'][0, 0]
G = constitutive_tens['E']['EGGbis']
D = constitutive_tens['EG']['EG'] - 2*np.vstack((G[:, :6], G[:, 6:]))
D_1_hom = D[0, 0]

U_hom = list()
for u_m, u_p in zip(u_m_sb, u_p_sb):
    expr = fe.Expression(
            f"x[0]<= A ? {sp.printing.ccode(u_m)} : {sp.printing.ccode(u_p)}",
            degree=6, F=load_magnitude/LY, A=load_x, L=LX,
            Q_1=Q_1_hom, D_1=D_1_hom)
    U_hom.append(expr)

# * Step 6 : Completing homogenized solution with micro-scale fluctuations
macro_fields = {
        'U': [U_hom[0], 0],
        'E': [U_hom[1], 0, 0],
        'EG': [U_hom[2]] + [0]*5,
        'EGG': [U_hom[3]] + [0]*11
    }

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
        //passer par un pointeur ? *f->eval(values, coord_equi);
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


# TODO Organisation du code à revoir
# TODO Peut-on s'éviter la compilation du code à chaque fois qu'une instance de PeriodicExpr est crée ?
# TODO la solution est peut-être dans un __init__ personnalisé :
# TODO Idée : reprendre le init de la classe mère sans l'écraser totalement et simplement lui rajouter la partie :
# TODO : __init(self, periodicity_vectors, local_function)
# TODO : self._cpp_object = fe.compile_cpp_code(per_scalar_fnct_cpp_code).PeriodicExpr()
# TODO self._cpp_object.per_x = per_x(periodicity_vectors) #per_x à définir, ajuster pour des cas où ce n'est pas un rectangle ?
# TODO self._cpp_object.per_y = per_y(periodicity_vectors)
class PeriodicExpr(fe.UserExpression):
    def value_shape(self):
        return ()
    # TODO : Regarder si on peut faire un __init__ particulier,
    # TODO tel qu'on conserve


# periodic_cppcode = fe.compile_cpp_code(per_scalar_fnct_cpp_code).PeriodicExpr() #? Peut-on le sortir de la boucle ci-dessous ?

key_conversion = {'U': 'u', 'Epsilon': 'eps', 'Sigma': 'sigma'}
localztn_expr = dict()
for key, scd_dict in hom_model.localization.items():
    if key == 'EGGbis':  # ! Je passe avec une clé EGG ?! Pourquoi ?
        continue
    localztn_expr[key] = dict()
    for scd_key, localztn_fields in scd_dict.items():
        updated_key2 = key_conversion[scd_key]
        localztn_expr[key][updated_key2] = list()  # 1 field per component of U, E, EG and EGG
        for i, field in enumerate(localztn_fields):
            new_fields = list()  # 1 field per component of U, Sigma and Epsilon
            for component in field.split():
                per_field = PeriodicExpr(degree=3)
                periodic_cppcode = fe.compile_cpp_code(per_scalar_fnct_cpp_code).PeriodicExpr()
                periodic_cppcode.per_x = rve_geo.gen_vect[0, 0]
                periodic_cppcode.per_y = rve_geo.gen_vect[1, 1]
                component.set_allow_extrapolation(True)
                periodic_cppcode.f = component.cpp_object()
                # https://bitbucket.org/fenics-project/dolfin/issues/1041/compiledexpression-cant-be-initialized
                per_field._cpp_object = periodic_cppcode
                new_fields.append(per_field)
            localztn_expr[key][updated_key2].append(new_fields)
function_spaces = {
    'u': model.displ_fspace,
    'eps': model.strain_fspace,
    }
scalar_fspace = model.scalar_fspace

fe.parameters["allow_extrapolation"] = True
reconstr_sol = full_scale_pb.reconstruction(
        localztn_expr, macro_fields, function_spaces, trunc_order=2, output_request=('eps', 'u'))

results_file = Path('demo_results/pantograph_compare.xdmf')
with fe.XDMFFile(str(results_file)) as f_out:
    data = [
        (reconstr_sol['u'], 'disp_reconstruction', 'displacement reconstruction'),
        (reconstr_sol['eps'], 'strain_reconstruction', 'strain reconstruction'),
        (fe.project(macro_fields['U'][0], scalar_fspace), 'disp_macro', 'disp_macro'),
        (fe.project(macro_fields['E'][0], scalar_fspace), 'strain_macro', 'strain_macro'),
        (model.u_sol, 'displ_exact', 'displ_exact'),
        (model.eps_sol, 'strain_exact', 'strain_exact'),
    ]
    f_out.parameters["flush_output"] = True
    f_out.parameters["functions_share_mesh"] = True
    for field, name, descrpt in data:
        field.rename(name, descrpt)
        f_out.write(field, 0.)

# * Step 7 : Calculation of the relative differences
exact_sol = {'u': model.u_sol, 'eps': model.eps_sol}
errors = dict()
for f_name in reconstr_sol.keys():
    dim = exact_sol[f_name].ufl_shape[0]
    exact_norm = fe.errornorm(
        exact_sol[f_name], fe.Constant([0.0]*dim),
        'L2', degree_rise=0, mesh=model.part.mesh)
    difference = fe.errornorm(
        reconstr_sol[f_name], exact_sol[f_name],
        'L2', degree_rise=0, mesh=model.part.mesh)
    error = difference/exact_norm
    errors[f_name] = error
    print(f_name, error, difference, exact_norm)
    logger.info(f'Relative error for {f_name} = {error}')
