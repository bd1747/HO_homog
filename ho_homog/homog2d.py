# coding: utf-8

import dolfin as fe
import logging
import numpy as np

from ho_homog import periodicity
from ho_homog.materials import epsilon, sigma
from ho_homog.toolbox_FEniCS import (DOLFIN_KRYLOV_METHODS, DOLFIN_LU_METHODS,
                                     local_project)

SOLVER_METHOD = 'mumps'

np.set_printoptions(precision=4, linewidth=150)
np.set_printoptions(suppress=True)

logging.getLogger('UFL').setLevel(logging.DEBUG)
logging.getLogger('FFC').setLevel(logging.DEBUG)
# https://fenicsproject.org/qa/3669/how-can-i-diable-log-and-debug-messages-from-ffc/
logger = logging.getLogger(__name__)


class Fenics2DHomogenization(object):
    ''' Homogenization of 2D orthotropic periodic laminates in plane strain. The direction of lamination is 3 and the invariant direction is 1
    '''
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
        self.topo_dim = topo_dim = fenics_2d_rve.mesh_dim
        self.pbc = periodicity.PeriodicDomain.pbc_dual_base(
            fenics_2d_rve.gen_vect, 'XY', topo_dim)

        solver = kwargs.pop('solver', {})
        # {'type': solver_type, 'method': solver_method, 'preconditioner': preconditioner}
        s_type = solver.pop('type', None)
        s_method = solver.pop('method', SOLVER_METHOD)
        s_precond = solver.pop('preconditioner', None)
        if s_type is None:
            if s_method in DOLFIN_KRYLOV_METHODS.keys():
                s_type = "Krylov"
            elif s_method in DOLFIN_LU_METHODS.keys():
                s_type = "LU"
            else:
                raise RuntimeError("The indicated solver method is unknown.")
        self._solver = dict(type=s_type, method=s_method)
        if s_precond:
            self._solver['preconditioner'] = s_precond

        element = kwargs.pop('element', ('Lagrange', 2))
        if isinstance(element, dict):
            element = (element['family'], element['degree'])
        self._element = element

        # * Function spaces
        cell = self.rve.mesh.ufl_cell()
        self.scalar_FE = fe.FiniteElement(element[0], cell, element[1])
        self.displ_FE = fe.VectorElement(element[0], cell, element[1])
        strain_deg = element[1] - 1 if element[1] >= 1 else 0
        strain_dim = int(topo_dim * (topo_dim + 1) / 2)
        self.strain_FE = fe.VectorElement('DG', cell, strain_deg,
                                          dim=strain_dim)
        # Espace fonctionel scalaire
        self.X = fe.FunctionSpace(self.rve.mesh, self.scalar_FE,
                                  constrained_domain=self.pbc)
        # Espace fonctionnel 3D : deformations, notations de Voigt
        self.W = fe.FunctionSpace(self.rve.mesh, self.strain_FE,
                                  constrained_domain=self.pbc)
        # Espace fonctionel 2D pour les champs de deplacement
        # TODO : reprendre le Ve défini pour l'espace fonctionnel mixte. Par ex: V = FunctionSpace(mesh, Ve)
        self.V = fe.VectorFunctionSpace(self.rve.mesh, element[0], element[1],
                                        constrained_domain=self.pbc)


        # * Espace fonctionel mixte pour la résolution : 2D pour les champs + scalaire pour multiplicateur de Lagrange

        # Pour le multiplicateur de Lagrange : Real element with one global degree of freedom
        self.real_FE = fe.VectorElement("R", cell, 0)
        self.M = fe.FunctionSpace(
            self.rve.mesh,
            fe.MixedElement([self.displ_FE, self.real_FE]),
            constrained_domain=self.pbc
            )

        # Define variational problem
        self.v, self.lamb_ = fe.TestFunctions(self.M)
        self.u, self.lamb = fe.TrialFunctions(self.M)
        self.w = fe.Function(self.M)

        # bilinear form
        self.a = (fe.inner(
                    sigma(self.rve.C_per, epsilon(self.u)),
                    epsilon(self.v)
                    ) * fe.dx
                  + fe.dot(self.lamb_, self.u)*fe.dx
                  + fe.dot(self.lamb, self.v)*fe.dx)
        self.K = fe.assemble(self.a)
        # ! self.solver = fe.LUSolver(self.K) #! Précédemment
        # ! Utilisé pour les études de convergence mais pas le plus approprié selon Jérémy
        # ! self.solver = fe.KrylovSolver(self.K, method="cg")
        self.solver = fe.LUSolver(self.K, "mumps")
        self.solver.parameters["symmetric"] = True
        # print("Solver parameters : ")
        # fe.info(self.solver.parameters, True)

        # Areas
        self.one = fe.interpolate(fe.Constant(1),self.X)

        self.localization = {}  # dictionary of localization field objects, to be filled up when calling auxiliary problems (lazy evaluation)
        self.ConstitutiveTensors = {}  # dictionary of homogenized constitutive tensors, to be filled up when calling auxiliary problems and averaging localization fields.

    def homogenizationScheme(self, model):
        DictOfLocalizationsU = {}
        DictOfLocalizationsSigma = {}
        DictOfLocalizationsEpsilon = {}
        DictOfConstitutiveTensors = {}

        DictOfLocalizationsU['U'] = self.LocalizationU()['U']

        if model == 'E' or model == 'EG' or model == 'EGG':
            DictOfLocalizationsU['E'] = self.LocalizationE()['U']
            DictOfLocalizationsSigma['E'] = self.LocalizationE()['Sigma']
            DictOfLocalizationsEpsilon['E'] = self.LocalizationE()['Epsilon']

        if model == 'EG':
            DictOfLocalizationsU['EGbis'] = self.LocalizationEG()['U']
            DictOfLocalizationsSigma['EGbis'] = self.LocalizationEGbis()['Sigma']
            DictOfLocalizationsEpsilon['EGbis'] = self.LocalizationEGbis()['Epsilon']

        if model == 'EGG':
            DictOfLocalizationsU['EG'] = self.LocalizationEG()['U']
            DictOfLocalizationsSigma['EG'] = self.LocalizationEG()['Sigma']
            DictOfLocalizationsEpsilon['EG'] = self.LocalizationEG()['Epsilon']

            DictOfLocalizationsU['EGGbis'] = self.LocalizationEGG()['U']
            DictOfLocalizationsSigma['EGGbis'] = self.LocalizationEGGbis()['Sigma']
            DictOfLocalizationsEpsilon['EGGbis'] = self.LocalizationEGGbis()['Epsilon']

        Keys = list(DictOfLocalizationsSigma.keys())

        for Key1 in Keys:
            DictOfConstitutiveTensors[Key1] = {}
            for Key2 in Keys:
                DictOfConstitutiveTensors[Key1][Key2] = 0.

        nk = len(Keys)
        for i in range(nk):
            Key1 = Keys[i]
            for j in range(i, nk):
                Key2 = Keys[j]
                C = self.CrossEnergy(Key1, Key2, DictOfLocalizationsSigma[Key1], DictOfLocalizationsEpsilon[Key2])
                DictOfConstitutiveTensors[Key1][Key2] = C
                DictOfConstitutiveTensors[Key2][Key1] = C.T

        return DictOfLocalizationsU, DictOfLocalizationsSigma, DictOfLocalizationsEpsilon, DictOfConstitutiveTensors

    def LocalizationU(self):
        try:
            out = self.localization['U']
        except KeyError:

            u = [fe.interpolate(fe.Constant((1., 0.)), self.V),
                 fe.interpolate(fe.Constant((0., 1.)), self.V)]
            s = [fe.interpolate(fe.Constant((0., 0., 0.)), self.W),
                 fe.interpolate(fe.Constant((0., 0., 0.)), self.W)]

            out = {'U': u, 'Sigma': s, 'Epsilon': s}
            self.localization['U'] = out
        return out

    def LocalizationE(self):
        '''
        return E localization fields (as function of macro E)


        '''
        try:
            out = self.localization['E']
        except KeyError:
            f = [fe.Constant((0, 0)), fe.Constant((0, 0)), fe.Constant((0, 0))]
            Fload = [fe.interpolate(fo, self.V) for fo in f]
            epsilon0 = [fe.Constant((1, 0, 0)), fe.Constant((0, 1, 0)), fe.Constant((0, 0, 1))]
            Epsilon0 = [fe.interpolate(eps, self.W) for eps in epsilon0]
            u, sigm, eps = self.genericAuxiliaryProblem(Fload, Epsilon0)
            out = {'U': u, 'Sigma': sigm, 'Epsilon': eps}
            self.localization['E'] = out
        return out

    def LocalizationEbis(self):
        '''
        return E stress/strain localization fields $u^U\diads\delta$ (as function of macro E)


        '''
        try:
            out = self.localization['Ebis']
        except KeyError:
            epsilon0 = [fe.Constant((1, 0, 0)), fe.Constant((0, 1, 0)), fe.Constant((0, 0, 1))]
            Epsilon0 = [fe.interpolate(eps, self.W) for eps in epsilon0]

            out = {
                'Sigma': [sigma(self.rve.C_per, Epsilon0[i]) for i in range(3)],
                'Epsilon': Epsilon0}
            self.localization['Ebis'] = out
        return out

    def LocalizationEG(self):
        '''
        return GradE localization fields (as function of macro GradE)

        '''
        try:
            out = self.localization['EG']
        except KeyError:
            self.LocalizationE()

            Fload = self.Sigma2Fload(self.localization['E']['Sigma'])
            Epsilon0 = self.Displacement2Epsilon0(self.localization['E']['U'])

            u, sigm, eps = self.genericAuxiliaryProblem(Fload, Epsilon0)
            out = {'U': u, 'Sigma': sigm, 'Epsilon': eps}
            self.localization['EG'] = out
        return out

    def LocalizationEGbis(self):
        '''
        return GradE stress/strain localization fields for the EG model $u^E\diads\delta$ (as function of macro GradE)

        '''
        try:
            out = self.localization['EGbis']
        except KeyError:
            # h = self.lamination.total_thickness
            self.LocalizationE()

            Epsilon0 = self.Displacement2Epsilon0(self.localization['E']['U'])

            out = {
                'Sigma': [sigma(self.rve.C_per, Epsilon0[i]) for i in range(6)],
                'Epsilon': Epsilon0}
            self.localization['EGbis'] = out
        return out

    def LocalizationEGG(self):
        '''
        return GradGradE localization fields (as function of macro GradGradE)

        >>> a = LaminatedComposite.paganosLaminate([1.0, 1.5, 1.0], [np.pi/3, -np.pi/3, np.pi/3])
        >>> Lam = Laminate2D(a)
        >>> plt=plotTransverseDistribution(Lam.LocalizationEGG()['U'])
        >>> show()
        '''
        try:
            out = self.localization['EGG']
        except KeyError:
            self.LocalizationEG()

            Fload = self.Sigma2Fload(self.localization['EG']['Sigma'])
            Epsilon0 = self.Displacement2Epsilon0(self.localization['EG']['U'])

            u, sigm, eps = self.genericAuxiliaryProblem(Fload, Epsilon0)

            out = {'U': u, 'Sigma': sigm, 'Epsilon': eps}
            self.localization['EGG'] = out
        return out

    def LocalizationEGGbis(self):
        '''
        return GradGradE stress/strain localization fields for the EGG model $u^EG\diads\delta$ (as function of macro GradGradE)

        >>> a = LaminatedComposite.paganosLaminate([1.0, 1.5, 1.0], [np.pi/3, -np.pi/3, np.pi/3])
        >>> Lam = Laminate2D(a)
        >>> plt=plotTransverseDistribution(Lam.LocalizationEGGbis()['Sigma'])
        >>> show()
        '''
        try:
            out = self.localization['EGGbis']
        except KeyError:
            # h = self.lamination.total_thickness
            self.LocalizationEG()

            Epsilon0 = self.Displacement2Epsilon0(self.localization['EG']['U'])

            out = {
                'Sigma': [sigma(self.rve.C_per, Epsilon0[i]) for i in range(12)],
                'Epsilon': Epsilon0}
            self.localization['EGGbis'] = out
        return out

    def LocalizationSG(self):
        '''
        return GradSigma localization fields (as function of macro GradSigma)

        '''
        try:
            out = self.localization['SG']
        except:
            self.LocalizationE()
            C = self.CrossEnergy('E','E',self.localization['E']['Sigma'], self.localization['E']['Epsilon'])
            BEps = self.localization['E']['Sigma']
            n = len(BEps)
            BSigma = [[BEps[i][j] * C for j in range(n)] for i in range(n)]
            # Assembling the constitutive matrix
            BSigma_i = []
            for i in range(n):
                BSigma_j = []
                for j in range(n):
                    B =fe.Function(self.X)
                    for k in range(n):
                        print(C[k, j])
                        print(BEps[i].sub(k))
                        B = fe.interpolate(B + BEps[i].sub(k) * fe.Constant(C[k,j]),self.X)
                    BSigma_j = BSigma_j + [B]
                BSigma_i = BSigma_i + [BSigma_j]

            Fload = self.Sigma2Fload(BSigma)
            epsilon0 = [fe.Constant((0, 0, 0))] * 6
            Epsilon0 = [fe.interpolate(eps, self.W) for eps in epsilon0]

            u, sigm, eps = self.genericAuxiliaryProblem(Fload, Epsilon0)

            out = {'U': u, 'Sigma': sigm, 'Epsilon': eps}
            self.localization['EG'] = out
        return out


    def CrossEnergy(self, Key1, Key2, Loc1, Loc2):
        ''' Calcul l'energie croisee des tenseurs de localisation Loc1 et Loc2
        Si Loc1 est un champ de contraintes alors Loc2 doit etre un champ de deformations (et inversement)
        '''
        try:
            C = self.ConstitutiveTensors[Key1][Key2]
        except KeyError:
            logger.info(f"Compute cross energy {Key1} and {Key2}")
            if Key1 == Key2:
                K = len(Loc1)
                C = np.zeros((K, K))
                for k in range(K):
                    for l in range(k, K):
                        C[k, l] = self.rve.StrainCrossEnergy(Loc1[k], Loc2[l])
                        C[l, k] = C[k, l]
            else:
                K = len(Loc1)
                L = len(Loc2)
                C = np.zeros((K, L))
                for k in range(K):
                    for l in range(L):
                        C[k, l] = self.rve.StrainCrossEnergy(Loc1[k], Loc2[l])
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
            logger.info("Progression : load %i / %i", i+1, len(Fload))
            L = (fe.dot(Fload[i], self.v)
                 + fe.inner(-sigma(self.rve.C_per, Epsilon0[i]), epsilon(self.v))
                 ) * fe.dx

            # u_s = fe.Function(self.V)
            res = fe.assemble(L)
            # self.bc.apply(res) #TODO à tester. Pas nécessaire pour le moment, la ligne # K,res = fe.assemble_system(self.a,L,self.bc) était commentée.
            # self.solver.solve(K, u_s.vector(), res) #* Previous method
            self.solver.solve(self.w.vector(), res)
            #* More info : https://fenicsproject.org/docs/dolfin/1.5.0/python/programmers-reference/cpp/la/PETScLUSolver.html

            (u_s, lamb) = fe.split(self.w)
            # Not need anymore.
            # u_av = [fe.assemble(u_s[k]*fe.dx)/self.rve.mat_area for k in range(d) ]
            # u_av = fe.interpolate(fe.Constant(u_av), self.V)
            # u_s = fe.project(u_s - u_av,self.V)
            # # u_s = u_s - u_av

            self.u_s = fe.project(u_s, self.V)  # ? Pas un autre moyen de le faire ?
            U2 = U2 + [self.u_s]
            E2 = E2 + [local_project(epsilon(u_s) + Epsilon0[i], self.W)]
            S2 = S2 + [local_project(sigma(self.rve.C_per, E2[i]), self.W)]
            # e2 = fe.Function(self.W)
            # e2.assign(self.RVE.epsilon(u_s) + Epsilon0[i])
            # E2 = E2 + [e2]

            # s2 = fe.Function(self.W)
            # s2.assign(self.RVE.sigma(E2[i]))
            # S2 = S2 + [s2]

        return U2, S2, E2

    def anyOrderAuxiliaryProblem(self, order=1):

        f = [fe.Constant((0, 0)), fe.Constant((0, 0)), fe.Constant((0, 0))]
        Fload = [fe.interpolate(fo, self.V) for fo in f]

        epsilon0 = [fe.Constant((1, 0, 0)), fe.Constant((0, 1, 0)), fe.Constant((0, 0, 1))]
        Epsilon0 = [fe.interpolate(eps, self.W) for eps in epsilon0]

        U = []
        Sigma = []
        Epsilon = []

        for i in range(order):
            logger.info(i)
            if i > 0:
                logger.info('compute load')
                Fload = self.Sigma2Fload(Sigma[i-1])
                Epsilon0 = self.Displacement2Epsilon0(U[i-1])

            u, sigm, eps = self.genericAuxiliaryProblem(Fload, Epsilon0)
            U = U + [u]
            Sigma = Sigma + [sigm]
            Epsilon = Epsilon + [eps]

        return U, Sigma, Epsilon

    def Sigma2Fload(self, S):
        '''genere le chargement volumique associe au localisateur en
        contraintes du probleme precedent'''
        Fload = []
        S_avout = []
        # Oter la valeur moyenne
        for i in range(len(S)):
            Sj = []
            for j in range(int(self.topo_dim*(self.topo_dim+1)/2)):
                S_av = fe.assemble(S[i][j]*fe.dx)/self.rve.mat_area
                S_av = fe.interpolate(fe.Constant(S_av), self.X)
                Sj = Sj + [S[i][j] - S_av]
            S_avout = S_avout +[Sj]
        # Allocation du chargement
        for i in range(len(S)):
            Fload = Fload + [fe.as_vector((S_avout[i][0],S_avout[i][2]))]

        for i in range(len(S)):
            Fload = Fload + [fe.as_vector((S_avout[i][2],S_avout[i][1]))]

        return Fload

    def Displacement2Epsilon0(self, U):
        '''Converti le localisateur en deplacement en champ de predeformation
        a appliquer au probleme auxiliaire suivant'''
        Epsilon0 = []

        for i in range(len(U)):
            Epsilon0 = Epsilon0 + [fe.as_vector((U[i][0], fe.interpolate(fe.Constant(0.),self.X), U[i][1]/fe.sqrt(2)))]

        for i in range(len(U)):
            Epsilon0 = Epsilon0 + [fe.as_vector((fe.interpolate(fe.Constant(0.),self.X), U[i][1], U[i][0]/fe.sqrt(2)))]

        return Epsilon0
