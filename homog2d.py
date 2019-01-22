# coding: utf-8

import copy

import dolfin as fe
import numpy as np
import logging
from logging.handlers import RotatingFileHandler

np.set_printoptions(precision=4,linewidth=150)
np.set_printoptions(suppress = True)

'''TODO:
    - implementer Stress gradient
'''

logger = logging.getLogger(__name__) #http://sametmax.com/ecrire-des-logs-en-python/
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(name)s :: %(message)s')
file_handler = RotatingFileHandler('activity.log', 'a', 1000000, 1)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler) #Pour écriture d'un fichier log
formatter = logging.Formatter('%(levelname)s :: %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler) 

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

            def inside(self, x, on_boundary):
                # return True if on left or bottom boundary AND NOT on one of the two slave edges
                #Left boundary
                Left = fe.near(float(x.dot(dualbasis[0])), 0.)
                # print(float(x.dot(dualbasis[0])))
                #Bottom Boundary
                Bottom = fe.near(float(x.dot(dualbasis[1])), 0.)
                #Slave Right
                SlaveR = fe.near(float((x - basis[0]).dot(dualbasis[0])), 0.)
                #Slave Top
                SlaveT = fe.near(float((x - basis[1]).dot(dualbasis[1])), 0.)
                
                return (Left or Bottom) and not(SlaveR or SlaveT) and on_boundary
        
            def map(self, x, y):
                #Slave Right
                SlaveR = fe.near(float((x - basis[0]).dot(dualbasis[0])), 0.)
                #Slave Top
                SlaveT = fe.near(float((x - basis[1]).dot(dualbasis[1])), 0.)
                
                if SlaveR and SlaveT:
                    for i in range(topo_dim):
                        y[i] = x[i] - basis[0][i] - basis[1][i]
    
                elif SlaveR:
                    for i in range(topo_dim):
                        y[i] = x[i] - basis[0][i]
                elif SlaveT:
                    for i in range(topo_dim):
                        y[i] = x[i] - basis[1][i]
                else:
                    for i in range(topo_dim):
                        y[i] = 1000 * (basis[1][i] + basis[0][i])
    
        self.pbc = PeriodicDomain()
        # Test PeriodicDomain
    #    coor = self.RVE.mesh.coordinates()
    #    for i in range(coor.shape[0]):
    #        print(coor[i],self.pbc.inside(coor[i],True))
        self.tol = 1e-12
        # ne semble pas etre pris en compte?? sans cette condition il arrive a inverser... et avec ca ne vaut pas 0 en 0
        def pinned_point(x, on_boundary):
            return on_boundary and x[0] < self.tol and x[1] < self.tol
    
    
        elemType = 'CG'
        order = 2
        #Espace fonctionnel 3D pour la representation de voigt des deformations
        self.T = fe.VectorElement(elemType, self.rve.mesh.ufl_cell(), order, dim=int(self.topo_dim*(self.topo_dim+1)/2))
        self.W = fe.FunctionSpace(self.rve.mesh,self.T, constrained_domain=self.pbc)

        # Espace fonctionel 1D 
        self.Q = fe.FiniteElement(elemType, self.rve.mesh.ufl_cell(), order)
        self.X = fe.FunctionSpace(self.rve.mesh,self.Q, constrained_domain=self.pbc)
        self.one = fe.interpolate(fe.Constant(1),self.X)
        
        # Espace fonctionel 2D pour le champ de deplacement
        self.V = fe.VectorFunctionSpace(self.rve.mesh, elemType, order, constrained_domain=self.pbc)
        self.bc = fe.DirichletBC(self.V, fe.Constant((0, 0)), pinned_point, method = 'pointwise')
        
        # #Espace fonctionnel 3D pour la representation de voigt des deformations
        # self.T = fe.VectorElement(elemType, self.rve.mesh.ufl_cell(), order, dim=int(self.topo_dim*(self.topo_dim+1)/2))
        # self.W = fe.FunctionSpace(self.rve.mesh,self.T, constrained_domain=self.pbc)
        
        # Define variational problem
        self.u = fe.TrialFunction(self.V)
        self.v = fe.TestFunction(self.V)

        # bilinear form
        self.a = fe.inner(self.rve.sigma(self.rve.epsilon(self.u)), self.rve.epsilon(self.v))*fe.dx
        self.K = fe.assemble(self.a)
        self.solver = fe.LUSolver(self.K)
        self.solver.parameters["symmetric"] = True
        
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
            print('load '+ str(i))
            L = fe.dot(Fload[i], self.v) * fe.dx + fe.inner(-self.rve.sigma(Epsilon0[i]), self.rve.epsilon(self.v)) * fe.dx
        
            u_s = fe.Function(self.V)
            res = fe.assemble(L)
            # self.bc.apply(res) #TODO à tester. Pas nécessaire pour le moment, la ligne # K,res = fe.assemble_system(self.a,L,self.bc) était commentée.
            # self.solver.solve(K, u_s.vector(), res) #* Previous method
            self.solver.solve(u_s.vector(), res)
            #* More info : https://fenicsproject.org/docs/dolfin/1.5.0/python/programmers-reference/cpp/la/PETScLUSolver.html

            u_av = [fe.assemble(u_s[k]*fe.dx)/self.rve.mat_area for k in range(d) ]
        
            u_av = fe.interpolate(fe.Constant(u_av), self.V)
            
            u_s = fe.project(u_s - u_av,self.V)
            # u_s = u_s - u_av
    
            U2 = U2 + [u_s]
            E2 = E2 + [fe.project(self.rve.epsilon(u_s) + Epsilon0[i],self.W)]
            S2 = S2 + [fe.project(self.rve.sigma(E2[i]),self.W)]
            # e2 = fe.Function(self.W)
            # e2.assign(self.RVE.epsilon(u_s) + Epsilon0[i])
            # E2 = E2 + [e2]
            
            # s2 = fe.Function(self.W)
            # s2.assign(self.RVE.sigma(E2[i]))
            # S2 = S2 + [s2]
 
        return U2,S2,E2
    
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
