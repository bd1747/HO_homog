# coding: utf-8

import dolfin as fe
import numpy as np
import logging

np.set_printoptions(precision=4,linewidth=150)
np.set_printoptions(suppress = True)

'''TODO:
    - implementer Stress gradient
'''

GEO_TOLERANCE = 1e-12

logger = logging.getLogger(__name__)

logging.getLogger('UFL').setLevel(logging.DEBUG)
logging.getLogger('FFC').setLevel(logging.DEBUG)
# https://fenicsproject.org/qa/3669/how-can-i-diable-log-and-debug-messages-from-ffc/



class Fenics2DHomogenization(object):
    ''' Homogenization of 2D orthotropic periodic laminates in plane strain. The direction of lamination is 3 and the invariant direction is 1
    '''
    def __init__(self, fenics_2d_rve):

        
        self.rve = fenics_2d_rve
        topo_dim = fenics_2d_rve.mesh_dim
        self.topo_dim = topo_dim
        self.basis = [fe.as_vector(fenics_2d_rve.gen_vect[:,i]) for i in range(topo_dim)]
        gv_dual = np.linalg.inv(fenics_2d_rve.gen_vect).T
        self.dualbasis = [fe.as_vector(gv_dual[:,i]) for i in range(topo_dim)]

        basis = self.basis
        dualbasis = self.dualbasis

        class PeriodicDomain(fe.SubDomain):
        #? Source : https://comet-fenics.readthedocs.io/en/latest/demo/periodic_homog_elas/periodic_homog_elas.html
            def __init__(self, tolerance=GEO_TOLERANCE):
                """ vertices stores the coordinates of the 4 unit cell corners"""
                fe.SubDomain.__init__(self, tolerance)
                self.tol = tolerance

            def inside(self, x, on_boundary):
                # return True if on left or bottom boundary AND NOT on one of the two slave edges
                #Left boundary
                Left = fe.near(float(x.dot(dualbasis[0])), 0., self.tol)
                #Bottom Boundary
                Bottom = fe.near(float(x.dot(dualbasis[1])), 0., self.tol)
                #Slave Right
                SlaveR = fe.near(float((x - basis[0]).dot(dualbasis[0])), 0., self.tol)
                #Slave Top
                SlaveT = fe.near(float((x - basis[1]).dot(dualbasis[1])), 0., self.tol)
                
                return (Left or Bottom) and not(SlaveR or SlaveT) and on_boundary
        
            def map(self, x, y):
                #Slave Right
                SlaveR = fe.near(float((x - basis[0]).dot(dualbasis[0])), 0., self.tol)
                #Slave Top
                SlaveT = fe.near(float((x - basis[1]).dot(dualbasis[1])), 0., self.tol)
                
                if SlaveR and SlaveT: # if on top-right corner
                    for i in range(topo_dim):
                        y[i] = x[i] - basis[0][i] - basis[1][i]
                elif SlaveR: # if on right boundary
                    for i in range(topo_dim):
                        y[i] = x[i] - basis[0][i]
                elif SlaveT:
                    for i in range(topo_dim):
                        y[i] = x[i] - basis[1][i]
                else:
                    for i in range(topo_dim):
                        y[i] = 1000 * (basis[1][i] + basis[0][i])
    
        self.pbc = PeriodicDomain()
    
        elemType = 'CG'
        order = 2
        #Espace fonctionnel 3D pour la representation de voigt des deformations
        self.T = fe.VectorElement(
            elemType, self.rve.mesh.ufl_cell(), order,
            dim=int(self.topo_dim*(self.topo_dim+1)/2)
            )
        self.W = fe.FunctionSpace(self.rve.mesh, self.T, constrained_domain=self.pbc)

        # Espace fonctionel 1D 
        self.Q = fe.FiniteElement(elemType, self.rve.mesh.ufl_cell(), order)
        self.X = fe.FunctionSpace(self.rve.mesh, self.Q, constrained_domain=self.pbc)
        
        # Espace fonctionel 2D pour les champs de deplacement
        self.V = fe.VectorFunctionSpace(
            self.rve.mesh, elemType, order,
            constrained_domain=self.pbc
            )
            #TODO : reprendre le Ve défini pour l'espace fonctionnel mixte. Par ex: V = FunctionSpace(mesh, Ve)
        
        #* Espace fonctionel mixte pour la résolution : 2D pour les champs + scalaire pour multiplicateur de Lagrange
        self.Ve = fe.VectorElement(elemType, self.rve.mesh.ufl_cell(), order)
        # Pour le multiplicateur de Lagrange : Real element with one global degree of freedom
        self.Re = fe.VectorElement("R", self.rve.mesh.ufl_cell(), 0)
        self.M = fe.FunctionSpace(
            self.rve.mesh,
            fe.MixedElement([self.Ve, self.Re]),
            constrained_domain=self.pbc
            )
        
        # Define variational problem
        self.v, self.lamb_ = fe.TestFunctions(self.M)
        self.u, self.lamb = fe.TrialFunctions(self.M)
        self.w = fe.Function(self.M)

        # bilinear form
        self.a = (fe.inner(
                    self.rve.sigma(self.rve.epsilon(self.u)),
                    self.rve.epsilon(self.v)
                    ) * fe.dx
                + fe.dot(self.lamb_, self.u)*fe.dx
                + fe.dot(self.lamb, self.v)*fe.dx)
        self.K = fe.assemble(self.a)
        #! self.solver = fe.LUSolver(self.K) #! Précédemment
        #! Utilisé pour les études de convergence mais pas le plus approprié selon Jérémy
        #! self.solver = fe.KrylovSolver(self.K, method="cg")
        self.solver = fe.LUSolver(self.K, "mumps")
        self.solver.parameters["symmetric"] = True
        # print("Solver parameters : ")
        # fe.info(self.solver.parameters, True)

        # Areas
        self.one = fe.interpolate(fe.Constant(1),self.X)

        self.localization = {}  # dictionary of localization field objects, to be filled up when calling auxiliary problems (lazy evaluation)
        self.ConstitutiveTensors = {}  # dictionary of homogenized constitutive tensors, to be filled up when calling auxiliary problems and averaging localization fields.

        
    def homogenizationScheme(self,model):
        DictOfLocalizationsU = {}
        DictOfLocalizationsSigma = {}
        DictOfLocalizationsEpsilon = {}
        DictOfConstitutiveTensors = {}
        
        DictOfLocalizationsU['U']     = self.LocalizationU()['U']
        
        if model == 'E' or model == 'EG' or model == 'EGG':
            DictOfLocalizationsU['E']      = self.LocalizationE()['U']
            DictOfLocalizationsSigma['E']  = self.LocalizationE()['Sigma']
            DictOfLocalizationsEpsilon['E']  = self.LocalizationE()['Epsilon']

            
        if model == 'EG': 
            DictOfLocalizationsU['EGbis']       = self.LocalizationEG()['U']
            DictOfLocalizationsSigma['EGbis']   = self.LocalizationEGbis()['Sigma']
            DictOfLocalizationsEpsilon['EGbis'] = self.LocalizationEGbis()['Epsilon']
            
        if model == 'EGG':
            DictOfLocalizationsU['EG']       = self.LocalizationEG()['U']
            DictOfLocalizationsSigma['EG']   = self.LocalizationEG()['Sigma']
            DictOfLocalizationsEpsilon['EG'] = self.LocalizationEG()['Epsilon']
             
            DictOfLocalizationsU['EGGbis']       = self.LocalizationEGG()['U']
            DictOfLocalizationsSigma['EGGbis']   = self.LocalizationEGGbis()['Sigma']
            DictOfLocalizationsEpsilon['EGGbis'] = self.LocalizationEGGbis()['Epsilon']
            
        # if model == 'MicroM':
        #     DictOfLocalizationsU['Phi']   = - self.LocalizationGradU3()['U']
        #     DictOfLocalizationsU['Chi']   = self.LocalizationK()['U']
        #     DictOfLocalizationsU['Theta'] = self.LocalizationKG()['U']
            
        #     DictOfLocalizationsSigma['Phi']   = self.LocalizationGradU3bis()['Sigma']
        #     DictOfLocalizationsSigma['Chi']   = self.LocalizationK()['Sigma']         - self.LocalizationKbis()['Sigma']
        #     DictOfLocalizationsSigma['Theta'] = self.LocalizationKG()['Sigma']        - self.LocalizationKGbis()['Sigma']
            
        #     DictOfLocalizationsSigma['GradU3']    = self.LocalizationGradU3bis()['Sigma']
        #     DictOfLocalizationsSigma['GradPhi']   = self.LocalizationKbis()['Sigma']
        #     DictOfLocalizationsSigma['GradChi']   = self.LocalizationKGbis()['Sigma']
        #     DictOfLocalizationsSigma['GradTheta'] = self.LocalizationKGGbis()['Sigma']
        
        
        
        
        Keys = list(DictOfLocalizationsSigma.keys())
        
        for Key1 in Keys:
            DictOfConstitutiveTensors[Key1] = {}
            for Key2 in Keys:
                DictOfConstitutiveTensors[Key1][Key2] = 0.

        nk = len(Keys)
        for i in range(nk):
            Key1 = Keys[i]
            for j in range(i,nk):
                Key2 = Keys[j]
                C = self.CrossEnergy(Key1, Key2, DictOfLocalizationsSigma[Key1], DictOfLocalizationsEpsilon[Key2])
                DictOfConstitutiveTensors[Key1][Key2] = C
                DictOfConstitutiveTensors[Key2][Key1] = C.T
                        
        return DictOfLocalizationsU, DictOfLocalizationsSigma, DictOfLocalizationsEpsilon, DictOfConstitutiveTensors
    
    def LocalizationU(self):
        try:
            out = self.localization['U']
        except:
            
            u = [fe.interpolate(fe.Constant((1.,0.)), self.V),
                 fe.interpolate(fe.Constant((0.,1.)), self.V)]
            s = [fe.interpolate(fe.Constant((0.,0.,0.)), self.W),
                 fe.interpolate(fe.Constant((0.,0.,0.)), self.W)]

            out = {'U':u, 'Sigma':s, 'Epsilon':s}
            self.localization['U'] = out
        return out
    
    def LocalizationE(self):
        '''
        return E localization fields (as function of macro E)


        '''
        try:
            out = self.localization['E']
        except:
            f = [fe.Constant((0,0)), fe.Constant((0,0)), fe.Constant((0,0))]
            Fload = [fe.interpolate(fo, self.V) for fo in f]
            epsilon0 = [fe.Constant((1,0,0)), fe.Constant((0,1,0)), fe.Constant((0,0,1))]
            Epsilon0 = [fe.interpolate(eps, self.W) for eps in epsilon0]
            u, sigma, epsilon = self.genericAuxiliaryProblem(Fload,Epsilon0)
            out = {'U':u, 'Sigma':sigma, 'Epsilon': epsilon}
            self.localization['E'] = out
        return out
    
    def LocalizationEbis(self):
        '''
        return E stress/strain localization fields $u^U\diads\delta$ (as function of macro E)
 

        '''
        try:
            out = self.localization['Ebis']
        except:
            epsilon0 = [fe.Constant((1,0,0)), fe.Constant((0,1,0)), fe.Constant((0,0,1))]
            Epsilon0 = [fe.interpolate(eps, self.W) for eps in epsilon0]
             
            out = {'Sigma':[self.rve.sigma(Epsilon0[i]) for i in range(3)],'Epsilon':Epsilon0}
            self.localization['Ebis'] = out
        return out

    def LocalizationEG(self):
        '''
        return GradE localization fields (as function of macro GradE)

        '''
        try:
            out = self.localization['EG']
        except:
            self.LocalizationE()
             
            Fload = self.Sigma2Fload(self.localization['E']['Sigma'])
            Epsilon0 = self.Displacement2Epsilon0(self.localization['E']['U']) 
                
            u, sigma, epsilon = self.genericAuxiliaryProblem(Fload,Epsilon0)
            out = {'U':u, 'Sigma':sigma, 'Epsilon': epsilon}
            self.localization['EG'] = out
        return out
     
    def LocalizationEGbis(self):
        '''
        return GradE stress/strain localization fields for the EG model $u^E\diads\delta$ (as function of macro GradE)
 
        '''
        try:
            out = self.localization['EGbis']
        except:
            # h = self.lamination.total_thickness
            self.LocalizationE()
             
            Epsilon0 = self.Displacement2Epsilon0(self.localization['E']['U'])
             
            out = {'Sigma':[self.rve.sigma(Epsilon0[i]) for i in range(6)],'Epsilon':Epsilon0}
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
        except:
            self.LocalizationEG()
             
            Fload = self.Sigma2Fload(self.localization['EG']['Sigma'])
            Epsilon0 = self.Displacement2Epsilon0(self.localization['EG']['U']) 
                
            u, sigma, epsilon = self.genericAuxiliaryProblem(Fload,Epsilon0)
             
            out = {'U':u, 'Sigma':sigma, 'Epsilon': epsilon}
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
        except:
#             h = self.lamination.total_thickness
            self.LocalizationEG()
             
            Epsilon0 = self.Displacement2Epsilon0(self.localization['EG']['U'])
             
            out = {'Sigma':[self.rve.sigma(Epsilon0[i]) for i in range(12)],'Epsilon':Epsilon0}
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
            C = self.CrossEnergy('E','E',self.localization['E']['Sigma'] ,self.localization['E']['Epsilon'])
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
                        print(C[k,j])
                        print(BEps[i].sub(k))
                        B = fe.interpolate(B + BEps[i].sub(k) * fe.Constant(C[k,j]),self.X)
                    BSigma_j = BSigma_j + [B]
                BSigma_i = BSigma_i + [BSigma_j]
            

            Fload = self.Sigma2Fload(BSigma)
            epsilon0 = [fe.Constant((0,0,0))] * 6
            Epsilon0 = [fe.interpolate(eps, self.W) for eps in epsilon0]
                            
            u, sigma, epsilon = self.genericAuxiliaryProblem(Fload,Epsilon0)
             
            out = {'U':u, 'Sigma':sigma, 'Epsilon': epsilon}
            self.localization['EG'] = out
        return out
        
    def CrossEnergy(self, Key1, Key2, Loc1, Loc2):
        ''' Calcul l'energie croisee des tenseurs de localisation Loc1 et Loc2
        Si Loc1 est un champ de contraintes alors Loc2 doit etre un champ de deformations (et inversement)
        '''
        try: 
            C = self.ConstitutiveTensors[Key1][Key2]
        except: 
            print('Compute cross energy '+Key1+' and '+Key2)
            if Key1==Key2:
                K = len(Loc1)
                C = np.zeros((K,K))
                for k in range(K):
                    for l in range(k,K):
                        # print(k,l)
                        C[k,l] = self.rve.StrainCrossEnergy(Loc1[k],Loc2[l])
                        C[l,k] = C[k,l]
            else:
                K = len(Loc1)
                L = len(Loc2)
                C = np.zeros((K,L))
                for k in range(K):
                    for l in range(L):
                        # print(k,l)
                        C[k,l] = self.rve.StrainCrossEnergy(Loc1[k],Loc2[l])
            try:
                self.ConstitutiveTensors[Key1][Key2] = C
            except:
                self.ConstitutiveTensors[Key1] = {}
                self.ConstitutiveTensors[Key1][Key2] = C 
                
            try:
                self.ConstitutiveTensors[Key2][Key1] = C.T
            except:
                self.ConstitutiveTensors[Key2] = {}
                self.ConstitutiveTensors[Key2][Key1] = C.T
        return C

    def genericAuxiliaryProblem(self,Fload,Epsilon0):
        '''This function compute the auxiliary problem of order N given the sources (of auxiliary problem N-1). It returns new localizations'''
        
        d = self.topo_dim

        U2 = []
        S2 = []
        E2 = []
        for i in range(len(Fload)):
            logger.info("Progression : load %i / %i", i+1, len(Fload))
            L = fe.dot(Fload[i], self.v) * fe.dx + fe.inner(-self.rve.sigma(Epsilon0[i]), self.rve.epsilon(self.v)) * fe.dx
        
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
            
            self.u_s = fe.project(u_s, self.V) #? Pas un autre moyen de le faire ?
            U2 = U2 + [self.u_s]
            E2 = E2 + [fe.project(self.rve.epsilon(u_s) + Epsilon0[i],self.W)]
            S2 = S2 + [fe.project(self.rve.sigma(E2[i]),self.W)]
            # e2 = fe.Function(self.W)
            # e2.assign(self.RVE.epsilon(u_s) + Epsilon0[i])
            # E2 = E2 + [e2]
            
            # s2 = fe.Function(self.W)
            # s2.assign(self.RVE.sigma(E2[i]))
            # S2 = S2 + [s2]
 
        return U2, S2, E2
    
    def anyOrderAuxiliaryProblem(self,order = 1):
        
        f = [fe.Constant((0,0)), fe.Constant((0,0)), fe.Constant((0,0))]
        Fload = [fe.interpolate(fo, self.V) for fo in f]
    
        epsilon0 = [fe.Constant((1,0,0)), fe.Constant((0,1,0)), fe.Constant((0,0,1))]
        Epsilon0 = [fe.interpolate(eps, self.W) for eps in epsilon0]
        
        U = []
        Sigma = []
        Epsilon = []
        
        for i in range(order):
            print(i)
            if i>0:
                print('compute load')
                Fload = self.Sigma2Fload(Sigma[i-1])
                Epsilon0 = self.Displacement2Epsilon0(U[i-1]) 
                
            u, sigma, epsilon = self.genericAuxiliaryProblem(Fload,Epsilon0)
            U = U + [u]
            Sigma = Sigma + [sigma]
            Epsilon = Epsilon + [epsilon]
            
        return U, Sigma, Epsilon
    
    def Sigma2Fload(self,S):
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

if __name__ == "__main__":
    import os
    import site
    CUR_DIR = os.path.dirname(os.path.realpath(__file__))
    site.addsitedir(CUR_DIR)
    #Pour que seulement ce qui est indiqué dans le .pth soit importable et non l'intégralité de ce dossier:
    #site.addpackage(CUR_DIR, '.pth', set())
    import mesh_generate_2D
    import materials
    import part
    import geometry
    import matplotlib.pyplot as plt
    from pathlib import Path

    geometry.init_geo_tools()

    logger_root = logging.getLogger()
    logger_root.setLevel(logging.INFO)
    formatter =logging.Formatter('%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s',"%Y-%m-%d %H:%M:%S")
    log_path = Path.home().joinpath('Desktop/activity.log')
    file_handler = RotatingFileHandler(str(log_path), 'a', 1000000, 10)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger_root.addHandler(file_handler) #Pour écriture d'un fichier log
    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s',"%H:%M")
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger_root.addHandler(stream_handler)
    
    E_NAMES = ('E11','E22','E12')

    def test_homogeneous_pantograph_cell():
        """ Test élémentaire.
        
        Homogénéisation d'une cellule homogène, construite avec la géométrie du 'pantographe'.

        """
        logger.info("Test : Homogeneization of a homogeneous cell with a pantograph geometry. Order : EGG")
        a = 1
        b, k = a, a/3
        junction_r = a/10
        geo_model = mesh_generate_2D.Gmsh2DRVE.pantograph(a, b, k, 0.1, soft_mat=True, name='homogeneous_panto')
        lc_ratio = 1/3
        d_min_max = (2*junction_r, a)
        geo_model.main_mesh_refinement(d_min_max, (lc_ratio*junction_r, lc_ratio*a), False)
        geo_model.soft_mesh_refinement(d_min_max, (lc_ratio*junction_r, lc_ratio*a), False)
        geo_model.mesh_generate()
        
        E, nu = 1., 0.3
        subdo_tags = tuple([subdo.tag for subdo in geo_model.phy_surf])
        material_dict = dict()
        for tag in subdo_tags:
            material_dict[tag] = materials.Material(E, nu, 'cp')
        
        rve = part.Fenics2DRVE.gmsh_2_Fenics_2DRVE(geo_model, material_dict,plots=False)
        hom_model = Fenics2DHomogenization(rve)
        results = hom_model.homogenizationScheme('E')
        localization_u, localization_sigma, localization_eps, constitutive_tens_dict = results
        print(constitutive_tens_dict['E']['E'])
        #* >> [[ 1.0989  0.3297 -0.    ]
        #*     [ 0.3297  1.0989 -0.    ]
        #*     [-0.     -0.      0.7692]]


        # print(constitutive_tens_dict['EG']['EG'])

        loc_E_u_file = fe.XDMFFile("homogeneous_cell_loc_E_u.xdmf")
        loc_E_u_file.parameters["flush_output"] = False
        loc_E_u_file.parameters["functions_share_mesh"] = True
        for field, E_name in zip(localization_u['E'], E_NAMES):
            print(localization_u['E'])
            print(type(localization_u['E']))
            print(type(localization_u['E'][0]))
            field.rename(f"loc_{E_name}_u", f"localization of displacement for {E_name}, homogeneous cell")
            loc_E_u_file.write(field, 0.)
        plt.figure()
        fe.plot(fe.project(0.1*localization_u['E'][2],hom_model.V), mode='displacement')
        plt.savefig("homogeneous_cell_loc_E12_u.pdf")
        return hom_model, results
    
    hom_model, results = test_homogeneous_pantograph_cell()
    plt.show()