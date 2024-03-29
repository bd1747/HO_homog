# coding: utf-8

import dolfin as fe
import logging
import numpy as np

from ho_homog import periodicity
from ho_homog.materials import epsilon, sigma, cross_energy
from ho_homog.toolbox_FEniCS import (
    DOLFIN_KRYLOV_METHODS,
    DOLFIN_LU_METHODS,
    local_project,
)

SOLVER_METHOD = "mumps"

U_NAMES = ("U1", "U2")
E_NAMES = ("E11", "E22", "E12")
EG_NAMES = ("EG111", "EG221", "EG121", "EG112", "EG222", "EG122")
EGG_NAMES = tuple(
    [n.replace("EG", "EGG") + "1" for n in EG_NAMES]
    + [n.replace("EG", "EGG") + "2" for n in EG_NAMES]
)

NAMES_MACRO_FIELDS = dict(
    U=U_NAMES,
    E=E_NAMES,
    EG=EG_NAMES,
    EGG=EGG_NAMES,
    EGbis=tuple([n.replace("EG", "EGbis") for n in EG_NAMES]),
    EGGbis=tuple([n.replace("EGG", "EGGbis") for n in EGG_NAMES]),
)


logging.getLogger("UFL").setLevel(logging.DEBUG)
logging.getLogger("FFC").setLevel(logging.DEBUG)
# https://fenicsproject.org/qa/3669/how-can-i-diable-log-and-debug-messages-from-ffc/
logger = logging.getLogger(__name__)


class Fenics2DHomogenization(object):
    """#TODOC"""

    def __init__(self, fenics_2d_rve, **kwargs):
        """[summary]

        Parameters
        ----------
        object : [type]
            [description]
        fenics_2d_rve : [type]
            [description]
        element : tuple or dict
            Type and degree of element for displacement FunctionSpace
            Ex: ('CG', 2) or {'family':'Lagrange', degree:2}
        solver : dict
            Choose the type of the solver, its method and the preconditioner.
            An up-to-date list of the available solvers and preconditioners
            can be obtained with dolfin.list_linear_solver_methods() and
            dolfin.list_krylov_solver_preconditioners().

        """
        self.rve = fenics_2d_rve
        self.topo_dim = topo_dim = fenics_2d_rve.dim
        try:
            bottom_left_corner = fenics_2d_rve.bottom_left_corner
        except AttributeError:
            logger.warning(
                "For the definition of the periodicity boundary conditions,"
                "the bottom left corner of the RVE is assumed to be on (0.,0.)"
            )
            bottom_left_corner = np.zeros(shape=(topo_dim,))
        self.pbc = periodicity.PeriodicDomain.pbc_dual_base(
            fenics_2d_rve.gen_vect, "XY", bottom_left_corner, topo_dim
        )

        solver = kwargs.pop("solver", {})
        # {'type': solver_type, 'method': solver_method, 'preconditioner': preconditioner}
        s_type = solver.pop("type", None)
        s_method = solver.pop("method", SOLVER_METHOD)
        s_precond = solver.pop("preconditioner", None)
        if s_type is None:
            if s_method in DOLFIN_KRYLOV_METHODS.keys():
                s_type = "Krylov"
            elif s_method in DOLFIN_LU_METHODS.keys():
                s_type = "LU"
            else:
                raise RuntimeError("The indicated solver method is unknown.")
        self._solver = dict(type=s_type, method=s_method)
        if s_precond:
            self._solver["preconditioner"] = s_precond

        element = kwargs.pop("element", ("Lagrange", 2))
        if isinstance(element, dict):
            element = (element["family"], element["degree"])
        self._element = element

        # * Function spaces
        cell = self.rve.mesh.ufl_cell()
        self.scalar_FE = fe.FiniteElement(element[0], cell, element[1])
        self.displ_FE = fe.VectorElement(element[0], cell, element[1])
        strain_deg = element[1] - 1 if element[1] >= 1 else 0
        strain_dim = int(topo_dim * (topo_dim + 1) / 2)
        self.strain_FE = fe.VectorElement("DG", cell, strain_deg, dim=strain_dim)
        # Espace fonctionel scalaire
        self.X = fe.FunctionSpace(
            self.rve.mesh, self.scalar_FE, constrained_domain=self.pbc
        )
        # Espace fonctionnel 3D : deformations, notations de Voigt
        self.W = fe.FunctionSpace(self.rve.mesh, self.strain_FE)
        # Espace fonctionel 2D pour les champs de deplacement
        # TODO : reprendre le Ve défini pour l'espace fonctionnel mixte. Par ex: V = FunctionSpace(mesh, Ve)
        self.V = fe.VectorFunctionSpace(
            self.rve.mesh, element[0], element[1], constrained_domain=self.pbc
        )

        # * Espace fonctionel mixte pour la résolution :
        # * 2D pour les champs + scalaire pour multiplicateur de Lagrange

        # "R" : Real element with one global degree of freedom
        self.real_FE = fe.VectorElement("R", cell, 0)
        self.M = fe.FunctionSpace(
            self.rve.mesh,
            fe.MixedElement([self.displ_FE, self.real_FE]),
            constrained_domain=self.pbc,
        )

        # Define variational problem
        self.v, self.lamb_ = fe.TestFunctions(self.M)
        self.u, self.lamb = fe.TrialFunctions(self.M)
        self.w = fe.Function(self.M)

        # bilinear form
        self.a = (
            fe.inner(sigma(self.rve.C_per, epsilon(self.u)), epsilon(self.v)) * fe.dx
            + fe.dot(self.lamb_, self.u) * fe.dx
            + fe.dot(self.lamb, self.v) * fe.dx
        )
        self.K = fe.assemble(self.a)
        if self._solver["type"] == "Krylov":
            self.solver = fe.KrylovSolver(self.K, self._solver["method"])
        elif self._solver["type"] == "LU":
            self.solver = fe.LUSolver(self.K, self._solver["method"])
            self.solver.parameters["symmetric"] = True
        try:
            self.solver.parameters.preconditioner = self._solver["preconditioner"]
        except KeyError:
            pass
        # fe.info(self.solver.parameters, True)

        self.localization = dict()
        # dictionary of localization field objects,
        # will be filled up when calling auxiliary problems (lazy evaluation)
        self.ConstitutiveTensors = dict()
        # dictionary of homogenized constitutive tensors,
        # will be filled up when calling auxiliary problems and
        # averaging localization fields.

    def homogenizationScheme(self, model):
        DictOfLocalizationsU = {}
        DictOfLocalizationsSigma = {}
        DictOfLocalizationsEpsilon = {}
        DictOfConstitutiveTensors = {}

        DictOfLocalizationsU["U"] = self.LocalizationU()["U"]

        if model == "E" or model == "EG" or model == "EGG":
            DictOfLocalizationsU["E"] = self.LocalizationE()["U"]

            DictOfLocalizationsSigma["E"] = self.LocalizationE()["Sigma"]
            DictOfLocalizationsEpsilon["E"] = self.LocalizationE()["Epsilon"]

            # TODO Renommer les champs enregistrés :
            #  for E_case, field in zip(E_NAMES, DictOfLocalizationsU["E"]):
            #   .rename(f"loc_{EG_case}_u", "")

        if model == "EG":
            DictOfLocalizationsU["EGbis"] = self.LocalizationEG()["U"]
            DictOfLocalizationsSigma["EGbis"] = self.LocalizationEGbis()["Sigma"]
            DictOfLocalizationsEpsilon["EGbis"] = self.LocalizationEGbis()["Epsilon"]

        if model == "EGG":
            DictOfLocalizationsU["EG"] = self.LocalizationEG()["U"]
            DictOfLocalizationsSigma["EG"] = self.LocalizationEG()["Sigma"]
            DictOfLocalizationsEpsilon["EG"] = self.LocalizationEG()["Epsilon"]

            DictOfLocalizationsU["EGGbis"] = self.LocalizationEGG()["U"]
            DictOfLocalizationsSigma["EGGbis"] = self.LocalizationEGGbis()["Sigma"]
            DictOfLocalizationsEpsilon["EGGbis"] = self.LocalizationEGGbis()["Epsilon"]

        # Prepare structure of DictOfConstitutiveTensors dictionary
        keys = list(DictOfLocalizationsSigma.keys())
        for k1 in keys:
            DictOfConstitutiveTensors[k1] = {}
            for k2 in keys:
                DictOfConstitutiveTensors[k1][k2] = None

        nk = len(keys)
        for i in range(nk):
            k1 = keys[i]
            for j in range(i, nk):
                k2 = keys[j]
                sig_fields_k1_cases = DictOfLocalizationsSigma[k1]
                eps_fields_k2_cases = DictOfLocalizationsEpsilon[k2]
                C = self.CrossEnergy(k1, k2, sig_fields_k1_cases, eps_fields_k2_cases)
                DictOfConstitutiveTensors[k1][k2] = C
                DictOfConstitutiveTensors[k2][k1] = C.T

        # Rename localization fields :
        for k in DictOfLocalizationsU.keys():
            for n, loc_f in zip(NAMES_MACRO_FIELDS[k], DictOfLocalizationsU[k]):
                loc_f.rename(f"loc_{n}_u", "")

        for k in DictOfLocalizationsSigma.keys():
            names = NAMES_MACRO_FIELDS[k]
            for i in range(len(DictOfLocalizationsSigma[k])):
                n = names[i]
                loc_f = DictOfLocalizationsSigma[k][i]
                try:
                    loc_f.rename(f"loc_{n}_sig", "")
                except AttributeError:
                    DictOfLocalizationsSigma[k][i] = local_project(loc_f, self.W)
                    DictOfLocalizationsSigma[k][i].rename(f"loc_{n}_sig", "")
                loc_f = DictOfLocalizationsEpsilon[k][i]
                try:
                    loc_f.rename(f"loc_{n}_eps", "")
                except AttributeError:
                    DictOfLocalizationsEpsilon[k][i] = local_project(loc_f, self.W)
                    DictOfLocalizationsEpsilon[k][i].rename(f"loc_{n}_eps", "")

        return (
            DictOfLocalizationsU,
            DictOfLocalizationsSigma,
            DictOfLocalizationsEpsilon,
            DictOfConstitutiveTensors,
        )

    def LocalizationU(self):
        try:
            out = self.localization["U"]
        except KeyError:

            u = [
                fe.interpolate(fe.Constant((1.0, 0.0)), self.V),
                fe.interpolate(fe.Constant((0.0, 1.0)), self.V),
            ]
            s = [
                fe.interpolate(fe.Constant((0.0, 0.0, 0.0)), self.W),
                fe.interpolate(fe.Constant((0.0, 0.0, 0.0)), self.W),
            ]

            out = {"U": u, "Sigma": s, "Epsilon": s}
            self.localization["U"] = out
        return out

    def LocalizationE(self):
        """
        return E localization fields (as function of macro E)


        """
        try:
            out = self.localization["E"]
        except KeyError:
            f = [fe.Constant((0, 0)), fe.Constant((0, 0)), fe.Constant((0, 0))]
            Fload = [fe.interpolate(fo, self.V) for fo in f]
            epsilon0 = [
                fe.Constant((1, 0, 0)),
                fe.Constant((0, 1, 0)),
                fe.Constant((0, 0, 1)),
            ]
            Epsilon0 = [fe.interpolate(eps, self.W) for eps in epsilon0]
            u, sigm, eps = self.genericAuxiliaryProblem(Fload, Epsilon0)
            out = {"U": u, "Sigma": sigm, "Epsilon": eps}
            self.localization["E"] = out
        return out

    def LocalizationEbis(self):
        """
        return E stress/strain localization fields $u^U\diads\delta$ (as function of macro E)


        """
        try:
            out = self.localization["Ebis"]
        except KeyError:
            epsilon0 = [
                fe.Constant((1, 0, 0)),
                fe.Constant((0, 1, 0)),
                fe.Constant((0, 0, 1)),
            ]
            Epsilon0 = [fe.interpolate(eps, self.W) for eps in epsilon0]

            out = {
                "Sigma": [sigma(self.rve.C_per, Epsilon0[i]) for i in range(3)],
                "Epsilon": Epsilon0,
            }
            self.localization["Ebis"] = out
        return out

    def LocalizationEG(self):
        """
        return GradE localization fields (as function of macro GradE)

        """
        try:
            out = self.localization["EG"]
        except KeyError:
            self.LocalizationE()

            Fload = self.Sigma2Fload(self.localization["E"]["Sigma"])
            Epsilon0 = self.Displacement2Epsilon0(self.localization["E"]["U"])

            u, sigm, eps = self.genericAuxiliaryProblem(Fload, Epsilon0)
            out = {"U": u, "Sigma": sigm, "Epsilon": eps}
            self.localization["EG"] = out
        return out

    def LocalizationEGbis(self):
        """
        return GradE stress/strain localization fields for the EG model $u^E\diads\delta$ (as function of macro GradE)

        """
        try:
            out = self.localization["EGbis"]
        except KeyError:
            # h = self.lamination.total_thickness
            self.LocalizationE()

            Epsilon0 = self.Displacement2Epsilon0(self.localization["E"]["U"])

            out = {
                "Sigma": [sigma(self.rve.C_per, Epsilon0[i]) for i in range(6)],
                "Epsilon": Epsilon0,
            }
            self.localization["EGbis"] = out
        return out

    def LocalizationEGG(self):
        """
        return GradGradE localization fields (as function of macro GradGradE)

        >>> a = LaminatedComposite.paganosLaminate([1.0, 1.5, 1.0], [np.pi/3, -np.pi/3, np.pi/3])
        >>> Lam = Laminate2D(a)
        >>> plt=plotTransverseDistribution(Lam.LocalizationEGG()['U'])
        >>> show()
        """
        try:
            out = self.localization["EGG"]
        except KeyError:
            self.LocalizationEG()

            Fload = self.Sigma2Fload(self.localization["EG"]["Sigma"])
            Epsilon0 = self.Displacement2Epsilon0(self.localization["EG"]["U"])

            u, sigm, eps = self.genericAuxiliaryProblem(Fload, Epsilon0)

            out = {"U": u, "Sigma": sigm, "Epsilon": eps}
            self.localization["EGG"] = out
        return out

    def LocalizationEGGbis(self):
        """
        return GradGradE stress/strain localization fields for the EGG model $u^EG\diads\delta$ (as function of macro GradGradE)

        >>> a = LaminatedComposite.paganosLaminate([1.0, 1.5, 1.0], [np.pi/3, -np.pi/3, np.pi/3])
        >>> Lam = Laminate2D(a)
        >>> plt=plotTransverseDistribution(Lam.LocalizationEGGbis()['Sigma'])
        >>> show()
        """
        try:
            out = self.localization["EGGbis"]
        except KeyError:
            # h = self.lamination.total_thickness
            self.LocalizationEG()

            Epsilon0 = self.Displacement2Epsilon0(self.localization["EG"]["U"])

            out = {
                "Sigma": [sigma(self.rve.C_per, Epsilon0[i]) for i in range(12)],
                "Epsilon": Epsilon0,
            }
            self.localization["EGGbis"] = out
        return out

    def LocalizationSG(self):
        """
        return GradSigma localization fields (as function of macro GradSigma)

        """
        try:
            out = self.localization["SG"]
        except KeyError:
            self.LocalizationE()
            C = self.CrossEnergy(
                "E",
                "E",
                self.localization["E"]["Sigma"],
                self.localization["E"]["Epsilon"],
            )
            BEps = self.localization["E"]["Sigma"]
            n = len(BEps)
            BSigma = [[BEps[i][j] * C for j in range(n)] for i in range(n)]
            # Assembling the constitutive matrix
            BSigma_i = []
            for i in range(n):
                BSigma_j = []
                for j in range(n):
                    B = fe.Function(self.X)
                    for k in range(n):
                        print(C[k, j])
                        print(BEps[i].sub(k))
                        B = fe.interpolate(
                            B + BEps[i].sub(k) * fe.Constant(C[k, j]), self.X
                        )
                    BSigma_j = BSigma_j + [B]
                BSigma_i = BSigma_i + [BSigma_j]

            Fload = self.Sigma2Fload(BSigma)
            epsilon0 = [fe.Constant((0, 0, 0))] * 6
            Epsilon0 = [fe.interpolate(eps, self.W) for eps in epsilon0]

            u, sigm, eps = self.genericAuxiliaryProblem(Fload, Epsilon0)

            out = {"U": u, "Sigma": sigm, "Epsilon": eps}
            self.localization["SG"] = out
        return out

    def CrossEnergy(self, Key1, Key2, Loc1, Loc2):
        """
        Calcul l'energie croisee des tenseurs de localisation Loc1 et Loc2.
        Si Loc1 est un champ de contraintes alors Loc2 doit etre un champ de
        deformations (et inversement)
        """
        try:
            C = self.ConstitutiveTensors[Key1][Key2]
        except KeyError:
            logger.info(f"Compute cross energy {Key1} and {Key2}")
            if Key1 == Key2:
                len_1 = len(Loc1)
                C = np.zeros((len_1, len_1))
                for k in range(len_1):
                    for l in range(k, len_1):
                        C[k, l] = (
                            cross_energy(Loc1[k], Loc2[l], self.rve.mesh)
                            / self.rve.rve_area
                        )
                        C[l, k] = C[k, l]
            else:
                len_1 = len(Loc1)
                len_2 = len(Loc2)
                C = np.zeros((len_1, len_2))
                for k in range(len_1):
                    for l in range(len_2):
                        C[k, l] = (
                            cross_energy(Loc1[k], Loc2[l], self.rve.mesh)
                            / self.rve.rve_area
                        )
            try:
                self.ConstitutiveTensors[Key1][Key2] = C
            except KeyError:
                self.ConstitutiveTensors[Key1] = {}
                self.ConstitutiveTensors[Key1][Key2] = C

            try:
                self.ConstitutiveTensors[Key2][Key1] = C.T
            except KeyError:
                self.ConstitutiveTensors[Key2] = {}
                self.ConstitutiveTensors[Key2][Key1] = C.T
        return C

    def genericAuxiliaryProblem(self, Fload, Epsilon0):
        """
        This function compute the auxiliary problem of order N
        given the sources (of auxiliary problem N-1).

        It returns new localizations
        """

        U2 = []
        S2 = []
        E2 = []
        for i in range(len(Fload)):
            logger.info(f"Progression : load {i+1} / {len(Fload)}")
            L = (
                fe.dot(Fload[i], self.v)
                + fe.inner(-sigma(self.rve.C_per, Epsilon0[i]), epsilon(self.v))
            ) * fe.dx

            # u_s = fe.Function(self.V)
            res = fe.assemble(L)
            # self.bc.apply(res) #TODO à tester. Pas nécessaire pour le moment, la ligne # K,res = fe.assemble_system(self.a,L,self.bc) était commentée.
            # self.solver.solve(K, u_s.vector(), res) #* Previous method
            self.solver.solve(self.w.vector(), res)
            # * More info : https://fenicsproject.org/docs/dolfin/1.5.0/python/programmers-reference/cpp/la/PETScLUSolver.html

            u_only = fe.interpolate(self.w.sub(0), self.V)
            # ? Essayer de faire fe.assign(self.u_only, self.w.sub(0)) ?
            # ? Pour le moment u_only vit dans V et V n'est pas extrait
            # ? de l'espace fonctionnel mixte. Est-ce que cela marcherait si V
            # ? est extrait de M ?
            U2.append(u_only)
            eps = epsilon(u_only) + Epsilon0[i]
            E2.append(local_project(eps, self.W))
            S2.append(local_project(sigma(self.rve.C_per, eps), self.W))
            # TODO : Comparer les options :
            # TODO         e2 = fe.Function(self.W); e2.assign(self.RVE.epsilon(u_s) + Epsilon0[i])
            # TODO         interpolation
            # TODO         projection (locale)

        return U2, S2, E2

    def anyOrderAuxiliaryProblem(self, order=1):

        f = [fe.Constant((0, 0)), fe.Constant((0, 0)), fe.Constant((0, 0))]
        Fload = [fe.interpolate(fo, self.V) for fo in f]

        epsilon0 = [
            fe.Constant((1, 0, 0)),
            fe.Constant((0, 1, 0)),
            fe.Constant((0, 0, 1)),
        ]
        Epsilon0 = [fe.interpolate(eps, self.W) for eps in epsilon0]

        U = []
        Sigma = []
        Epsilon = []

        for i in range(order):
            logger.info(i)
            if i > 0:
                logger.info("compute load")
                Fload = self.Sigma2Fload(Sigma[i - 1])
                Epsilon0 = self.Displacement2Epsilon0(U[i - 1])

            u, sigm, eps = self.genericAuxiliaryProblem(Fload, Epsilon0)
            U = U + [u]
            Sigma = Sigma + [sigm]
            Epsilon = Epsilon + [eps]

        return U, Sigma, Epsilon

    def Sigma2Fload(self, S):
        """genere le chargement volumique associe au localisateur en
        contraintes du probleme precedent"""
        Fload = []
        S_avout = []
        # Oter la valeur moyenne
        for i in range(len(S)):
            Sj = []
            for j in range(int(self.topo_dim * (self.topo_dim + 1) / 2)):
                S_av = fe.assemble(S[i][j] * fe.dx) / self.rve.mat_area
                S_av = fe.interpolate(fe.Constant(S_av), self.X)
                Sj = Sj + [S[i][j] - S_av]
            S_avout = S_avout + [Sj]
        # Allocation du chargement
        for i in range(len(S)):
            Fload = Fload + [fe.as_vector((S_avout[i][0], S_avout[i][2]))]

        for i in range(len(S)):
            Fload = Fload + [fe.as_vector((S_avout[i][2], S_avout[i][1]))]

        return Fload

    def Displacement2Epsilon0(self, U):
        """Converti le localisateur en deplacement en champ de predeformation
        a appliquer au probleme auxiliaire suivant"""
        Epsilon0 = []

        for i in range(len(U)):
            Epsilon0 = Epsilon0 + [
                fe.as_vector(
                    (
                        U[i][0],
                        fe.interpolate(fe.Constant(0.0), self.X),
                        U[i][1] / fe.sqrt(2),
                    )
                )
            ]
        # zero_ = fe.interpolate(fe.Constant(0.0), self.X)
        # # Possibilité de mettre directement fe.Constant(0.0) ?
        # prestrain_ = (U[i][0], zero_, U[i][1] / fe.sqrt(2))
        # prestrain_ = fe.as_vector(prestrain_)
        # Epsilon0.append(prestrain_)

        for i in range(len(U)):
            Epsilon0 = Epsilon0 + [
                fe.as_vector(
                    (
                        fe.interpolate(fe.Constant(0.0), self.X),
                        U[i][1],
                        U[i][0] / fe.sqrt(2),
                    )
                )
            ]
        # zero_ = fe.interpolate(fe.Constant(0.0), self.X)
        # # Possibilité de mettre directement fe.Constant(0.0) ?
        # prestrain_ = (zero_, U[i][1], U[i][0] / fe.sqrt(2))
        # prestrain_ = fe.as_vector(prestrain_)
        # Epsilon0.append(prestrain_)
        return Epsilon0
