# -*- coding: utf-8 -*-



from .GP import train_gp
#from .GP_lin import train_gp
from .utils import latin_hypercube,to_size
from .LBFGS_torch import acq_min_msp
from .SIR import SIR

import numpy as np
import torch
from scipy.spatial import KDTree,Voronoi
from math import sqrt, log



class VGT():
    def __init__(self, f, lb, ub, n_init, max_iter, Cp=0.1, use_approximation = False, N_neighbor = None, use_subspace = False, SIR_parameter = (4,3), _Rp = 4, num_samples = 20000, N_bars = None):
        self.lb = lb
        self.ub = ub
        self.dim = len(lb)
        self.f = f
        self.max_iter = max_iter
        
        self.Cp = Cp
        self.num_samples = num_samples
        
        self.use_subspace = use_subspace
        
        self.use_approximation = False if self.dim<=6 and not use_approximation else True
        
        if self.use_approximation:
            if N_neighbor is None:
                self.N_neighbor = max(1,self.dim // 2)
            else:
                self.N_neighbor = N_neighbor
        
            if N_bars is None:
                self.N_bars = max(min(1200,3*self.dim),self.dim+1)
            else:
                self.N_bars = N_bars

            self.adjacent_matrix = np.zeros((max_iter,max_iter))
        #self.bar_matrix = np.zeros((max_iter,max_iter))
        
            self.dist_matrix = np.zeros((max_iter,max_iter)) #np.inf * np.ones((max_iter,max_iter))
        self.x_opt = None
        self.y_opt = None
        self.current_pnts = 0
        self.nodes = []
        self.device = 'cpu'
        
        if use_subspace:
            
            assert isinstance(SIR_parameter[0],int) and SIR_parameter[0]>0,"The 'K' parameter of SIR must be positive integer."
            assert isinstance(SIR_parameter[1],int) and SIR_parameter[1]>0,"The 'H' parameter of SIR must be positive integer."
            assert SIR_parameter[1]<SIR_parameter[0],"The 'K' parameter of SIR must be smaller than 'H'."
            assert SIR_parameter[0]<N_neighbor,"Slice number of SIR must be smaller than the number of neighbors."
            self.n_feature = SIR_parameter[1]
            self.SIR = SIR(K = self.n_feature, H = SIR_parameter[0])
            assert _Rp >= 1,"Please set _Rp>=1 to ensure better results."
            self.Rp = int(1+_Rp)
        self.n_init = n_init
        self.sigma_x = 1e-3
        
        self.dk = None
        
        #self.random_sample(n_init)
        
        
        
    def random_sample(self,n_init):
        self.target_cell = None 
        x_sample = to_size(latin_hypercube(n_init, self.dim), self.lb, self.ub)
        y = self.f(x_sample)
        self.opt_idx = np.argmin(y)
        self.y_opt = y[self.opt_idx]
        self.x_opt = x_sample[self.opt_idx]
        
        self.current_pnts = n_init-1   
        
        if self.use_approximation:
            for j in range(n_init):
                for i in range(j):
                    self.dist_matrix[j,i] = self.dist_matrix[i, j] = np.abs(x_sample[j]-x_sample[i]).sum()
        
        self.build_graph()
        
        for j in range(n_init):
            newnode = node(x_sample[j,:],y[j],j,n_init,self.Cp)
            self.nodes.append(newnode)
        
        self.select_cell()
        
        return 
        
    def collect_sample(self,x):
        y = self.f(x)[0]
        self.current_pnts += 1
        
        if self.use_approximation:
            for j in range(self.current_pnts):
                self.dist_matrix[j,self.current_pnts] = self.dist_matrix[self.current_pnts, j] = np.abs(self.nodes[j].x-x).sum()
        if self.y_opt > y:
            self.y_opt = y
            self.dk = x - self.x_opt
            self.x_opt = x
            self.opt_idx = self.current_pnts
        self.build_graph()
        
        
        newnode = node(x,y,self.current_pnts,self.current_pnts+1,self.Cp)
        self.nodes.append(newnode)
        
        return
    

    
    def select_cell(self):
        
        if self.target_cell is None:
            self.target_cell = self.opt_idx
        else:
            idx = self.target_cell
            best_Q = self.nodes[idx].get_Q(self.current_pnts)
            if self.use_approximation:
                adj = self.adjacent_matrix[idx]
            else:
                adj = np.zeros(self.current_pnts+1,dtype=int)
                adj[self.neighbors] = 1
            for j in range(self.current_pnts+1):
                if adj[j] == 1:
                    if self.nodes[j].get_Q(self.current_pnts) > best_Q:
                        idx = j
                        best_Q = self.nodes[j].Q
            self.target_cell = idx
            
        if not self.use_approximation:
            self.find_neighbors()
        print('selected cell f(x) = ',self.nodes[self.target_cell].y)
        return self.target_cell 

    
    def propose_sample_PI_k(self, num_samples):
        
        xs = np.array([s.x for s in self.nodes])
        ys = np.array([s.y for s in self.nodes])
        if self.use_approximation:
            ranks = np.argsort(self.dist_matrix[self.target_cell,:self.current_pnts+1])
            adjacent_cells = ranks[:2*self.N_neighbor]
            self.adjacent_cells = np.zeros(self.current_pnts+1)>1
            self.adjacent_cells[adjacent_cells] = True
            #self.adjacent_matrix[self.target_cell]==1
            self.adjacent_cells[self.target_cell] = True
            
            x_train = xs[self.adjacent_cells[:self.current_pnts+1]]
            y_train = ys[self.adjacent_cells[:self.current_pnts+1]]
            #print(y_train)
            #print('number of points to train GP:',len(y_train))
            
            bar_cells_idx = ranks[:self.N_bars]
            bar_cells_tag = np.zeros(self.current_pnts+1)>1
            bar_cells_tag[bar_cells_idx] = True
            
            #print(np.arange(self.current_pnts+1)[bar_cells_tag]==self.target_cell)
            target_idx = np.where(np.arange(self.current_pnts+1)[bar_cells_tag]==self.target_cell)[0][0]
            xs_kdtree = xs[bar_cells_tag[:self.current_pnts+1]]
            #print('num of neighbors = ',len(xs_kdtree))
            
        else:
            x_train = xs[self.neighbors]
            y_train = ys[self.neighbors]
            xs_kdtree = xs[self.neighbors]
            target_idx = 0
        
        self.GP = train_gp(torch.tensor(x_train,device = self.device), torch.tensor(y_train,device = self.device),use_ard=False, num_steps=200)
        self.kdtree = KDTree(xs_kdtree)
        
        r_init = 1.0 * np.std(x_train,axis = 0) + self.sigma_x
        #print('r = ',r_init.min(),r_init.max())
        
        
        def pi(Xin):
            x = torch.atleast_2d(Xin)
            vals = self.GP.PI_nograd(x)
            outcell = self.kdtree.query(x.detach().numpy(), eps=0, k=1)[1] != target_idx 
            vals[outcell] = -1 #* torch.ones()
            return vals
        
        r = r_init
        num_sample = num_samples//10
        x_init = self.nodes[self.target_cell].x
        #lcb_min = self.GP.LCB_nograd(torch.tensor(x_init,device=self.device))
        
        for j in range(10):
            target_region = np.array([np.maximum(self.nodes[self.target_cell].x-r,self.lb), np.minimum(self.nodes[self.target_cell].x+r,self.ub)])
            
            x_sample = to_size(latin_hypercube(num_sample, self.dim), target_region[0], target_region[1])
            pi_val = -1.0 * np.ones(num_sample)
            incell = self.kdtree.query(x_sample, eps=0, k=1)[1] == target_idx
            pi_val[incell] = self.GP.PI_nograd(torch.tensor(x_sample[incell],device=self.device)).detach().numpy()
            #if lcb_val.min() < 0 and lcb_val.min() < lcb_min:
            if pi_val.max() >= 0.0:
                x_init = x_sample[np.argmax(pi_val)]
                #print('PI_init = ',pi_val.max())
                break
            else:
                r = 0.7 * r
        
        xopt, yopt = acq_min_msp(lambda x:-pi(x), lambda x:-finite_diff(x,pi), torch.tensor(x_init,device = self.device), torch.tensor(target_region,device = self.device), n_warmup=10000)
        
        #print('PI = ',-yopt)
        
        return xopt.detach().numpy()
    
    def propose_sample_EI_k(self, num_samples):
        
        xs = np.array([s.x for s in self.nodes])
        ys = np.array([s.y for s in self.nodes])
        if self.use_approximation:
            ranks = np.argsort(self.dist_matrix[self.target_cell,:self.current_pnts+1])
            adjacent_cells = ranks[:2*self.N_neighbor]
            self.adjacent_cells = np.zeros(self.current_pnts+1)>1
            self.adjacent_cells[adjacent_cells] = True
            #self.adjacent_matrix[self.target_cell]==1
            self.adjacent_cells[self.target_cell] = True
            
            x_train = xs[self.adjacent_cells[:self.current_pnts+1]]
            y_train = ys[self.adjacent_cells[:self.current_pnts+1]]
            #print(y_train)
            #print('number of points to train GP:',len(y_train))
            
            bar_cells_idx = ranks[:self.N_bars]
            bar_cells_tag = np.zeros(self.current_pnts+1)>1
            bar_cells_tag[bar_cells_idx] = True
            
            #print(np.arange(self.current_pnts+1)[bar_cells_tag]==self.target_cell)
            target_idx = np.where(np.arange(self.current_pnts+1)[bar_cells_tag]==self.target_cell)[0][0]
            xs_kdtree = xs[bar_cells_tag[:self.current_pnts+1]]
            #print('num of neighbors = ',len(xs_kdtree))
            
        else:
            x_train = xs[self.neighbors]
            y_train = ys[self.neighbors]
            xs_kdtree = xs[self.neighbors]
            target_idx = 0
        
        self.GP = train_gp(torch.tensor(x_train,device = self.device), torch.tensor(y_train,device = self.device),use_ard=False, num_steps=200)
        self.kdtree = KDTree(xs_kdtree)
        
        r_init = 1.0 * np.std(x_train,axis = 0) + self.sigma_x
        #print('r = ',r_init.min(),r_init.max())
        
        
        def ei(Xin):
            x = torch.atleast_2d(Xin)
            vals = self.GP.EI_nograd(x)
            outcell = self.kdtree.query(x.detach().numpy(), eps=0, k=1)[1] != target_idx 
            vals[outcell] = -1 #* torch.ones()
            return vals
        
        r = r_init
        num_sample = num_samples//10
        x_init = self.nodes[self.target_cell].x
        #lcb_min = self.GP.LCB_nograd(torch.tensor(x_init,device=self.device))
        
        for j in range(10):
            target_region = np.array([np.maximum(self.nodes[self.target_cell].x-r,self.lb), np.minimum(self.nodes[self.target_cell].x+r,self.ub)])
            
            x_sample = to_size(latin_hypercube(num_sample, self.dim), target_region[0], target_region[1])
            ei_val = -1.0 * np.ones(num_sample)
            incell = self.kdtree.query(x_sample, eps=0, k=1)[1] == target_idx
            ei_val[incell] = self.GP.EI_nograd(torch.tensor(x_sample[incell],device=self.device)).detach().numpy()
            #if lcb_val.min() < 0 and lcb_val.min() < lcb_min:
            if ei_val.max() >= 0.:
                x_init = x_sample[np.argmax(ei_val)]
                #print('EI_init = ',ei_val.max())
                break
            else:
                r = 0.8 * r
        
        xopt, yopt = acq_min_msp(lambda x:-ei(x), lambda x:-finite_diff(x,ei), torch.tensor(x_init,device = self.device), torch.tensor(target_region,device = self.device), n_warmup=10000)
        
        #print('EI = ',-yopt)
        
        return xopt.detach().numpy()
    
    def propose_sample_LCB_k(self, num_samples):
        
        xs = np.array([s.x for s in self.nodes])
        ys = np.array([s.y for s in self.nodes])
        if self.use_approximation:
            ranks = np.argsort(self.dist_matrix[self.target_cell,:self.current_pnts+1])
            adjacent_cells = ranks[:2*self.N_neighbor]
            self.adjacent_cells = np.zeros(self.current_pnts+1)>1
            self.adjacent_cells[adjacent_cells] = True
            #self.adjacent_matrix[self.target_cell]==1
            self.adjacent_cells[self.target_cell] = True
            
            x_train = xs[self.adjacent_cells[:self.current_pnts+1]]
            y_train = ys[self.adjacent_cells[:self.current_pnts+1]]
            #print(y_train)
            #print('number of points to train GP:',len(y_train))
            
            bar_cells_idx = ranks[:self.N_bars]
            bar_cells_tag = np.zeros(self.current_pnts+1)>1
            bar_cells_tag[bar_cells_idx] = True
            
            #print(np.arange(self.current_pnts+1)[bar_cells_tag]==self.target_cell)
            target_idx = np.where(np.arange(self.current_pnts+1)[bar_cells_tag]==self.target_cell)[0][0]
            xs_kdtree = xs[bar_cells_tag[:self.current_pnts+1]]
            #print('num of neighbors = ',len(xs_kdtree))
            
        else:
            x_train = xs[self.neighbors]
            y_train = ys[self.neighbors]
            xs_kdtree = xs[self.neighbors]
            target_idx = 0
        
        self.GP = train_gp(torch.tensor(x_train,device = self.device), torch.tensor(y_train,device = self.device),use_ard=False, num_steps=200)
        self.kdtree = KDTree(xs_kdtree)
        
        r_init = 1.0 * np.std(x_train,axis = 0) + self.sigma_x
        #print('r = ',r_init.min(),r_init.max())
        
        
        def lcb(Xin):
            x = torch.atleast_2d(Xin)
            vals = self.GP.LCB_nograd(x)
            outcell = self.kdtree.query(x.detach().numpy(), eps=0, k=1)[1] != target_idx 
            vals[outcell] = 5 #* torch.ones()
            return vals
        
        r = r_init
        num_sample = num_samples//10
        x_init = self.nodes[self.target_cell].x
        #lcb_min = self.GP.LCB_nograd(torch.tensor(x_init,device=self.device))
        
        for j in range(10):
            target_region = np.array([np.maximum(self.nodes[self.target_cell].x-r,self.lb), np.minimum(self.nodes[self.target_cell].x+r,self.ub)])
            
            x_sample = to_size(latin_hypercube(num_sample, self.dim), target_region[0], target_region[1])
            lcb_val = 5.0 * np.ones(num_sample)
            incell = self.kdtree.query(x_sample, eps=0, k=1)[1] == target_idx
            lcb_val[incell] = self.GP.LCB_nograd(torch.tensor(x_sample[incell],device=self.device)).detach().numpy()
            #if lcb_val.min() < 0 and lcb_val.min() < lcb_min:
            if lcb_val.min() < 0.:
                x_init = x_sample[np.argmax(lcb_val)]
                #print('LCB_init = ',lcb_val.min())
                break
            else:
                r = 0.8 * r
        
        xopt, yopt = acq_min_msp(lambda x:lcb(x), lambda x:finite_diff(x,lcb), torch.tensor(x_init,device = self.device), torch.tensor(target_region,device = self.device), n_warmup=10000)
        
        #print('LCB = ',yopt)
        
        return xopt.detach().numpy()
    
    
    
    def propose_sample_pi(self, num_samples):
        
        self.adjacent_cells = self.adjacent_matrix[self.target_cell]==1
        self.adjacent_cells[self.target_cell] = True
        xs = np.array([s.x for s in self.nodes])
        ys = np.array([s.y for s in self.nodes])
        x_train = xs[self.adjacent_cells[:self.current_pnts+1]]
        y_train = ys[self.adjacent_cells[:self.current_pnts+1]]
        #print(y_train)
        #print('number of points to train GP:',len(y_train))
        #print()
        self.GP = train_gp(torch.tensor(x_train,device = self.device), torch.tensor(y_train,device = self.device),use_ard=False, num_steps=200)
        self.kdtree = KDTree(xs)
        
        r_init = 1.0 * np.std(x_train,axis = 0) + self.sigma_x
        #print('r = ',r_init.min(),r_init.max())
        
        
        def pi(Xin):
            x = torch.atleast_2d(Xin)
            vals = self.GP.PI_nograd(x)
            outcell = self.kdtree.query(x.detach().numpy(), eps=0, k=1)[1] != self.target_cell 
            vals[outcell] = 5 #* torch.ones()
            return vals
        
        r = r_init
        num_sample = num_samples//10
        x_init = self.nodes[self.target_cell].x
        #lcb_min = self.GP.LCB_nograd(torch.tensor(x_init,device=self.device))
        
        for j in range(10):
            target_region = np.array([np.maximum(self.nodes[self.target_cell].x-r,self.lb), np.minimum(self.nodes[self.target_cell].x+r,self.ub)])
            
            x_sample = to_size(latin_hypercube(num_sample, self.dim), target_region[0], target_region[1])
            lcb_val = 5 * np.ones(num_sample)
            incell = self.kdtree.query(x_sample, eps=0, k=1)[1] == self.target_cell
            lcb_val[incell] = self.GP.LCB_nograd(torch.tensor(x_sample[incell],device=self.device)).detach().numpy()
            #if lcb_val.min() < 0 and lcb_val.min() < lcb_min:
            if lcb_val.min() < 0:
                x_init = x_sample[np.argmin(lcb_val)]
                #print('lcb_init = ',lcb_val.min())
                break
            else:
                r = 0.8 * r
        
        xopt, yopt = acq_min_msp(lambda x:-pi(x), lambda x:-finite_diff(x,pi), torch.tensor(x_init,device = self.device), torch.tensor(target_region,device = self.device), n_warmup=10000)
        
        #print('lcb = ',yopt)
        
        return xopt.detach().numpy()
    
    def propose_sample_subspace_bo(self, num_samples):
        
        self.adjacent_cells = self.adjacent_matrix[self.target_cell]==1
        self.adjacent_cells[self.target_cell] = True
        xs = np.array([s.x for s in self.nodes])
        ys = np.array([s.y for s in self.nodes])
        x_train = xs[self.adjacent_cells[:self.current_pnts+1]]
        y_train = ys[self.adjacent_cells[:self.current_pnts+1]]
        #print(y_train)
        #sprint('number of points to train subspace GP:',len(y_train))
        #print()
        
        self.kdtree = KDTree(xs)
        
        r0 = 1.0 * np.std(x_train,axis = 0) + self.sigma_x
        sensitive_dims = r0 > 2 * self.sigma_x 
        
        self.SIR.fit(x_train[:,sensitive_dims],y_train)
        #print('beta = ', self.SIR.beta,x_train[:,sensitive_dims])
        
        if self.dk is None:
            edr = np.zeros((self.n_feature,self.dim))
            edr[:self.n_feature, sensitive_dims] = self.SIR.beta.real.T
        else:
            features = self.SIR.beta.real.T
            edr = np.zeros((features.shape[0]+1,self.dim))
            edr[0] = self.dk
            edr[1:features.shape[0]+1, sensitive_dims] = features
            
        edrs = orthogonalize(edr)
        sub_dim = edrs.shape[0]
        x0 = self.nodes[self.target_cell].x
        
        self.GP = train_gp(torch.tensor(x_train,device = self.device), torch.tensor(y_train,device = self.device),use_ard=False, num_steps=200)
        
        
        r_init = np.linalg.norm(r0)
        
        def lcb(s_in):
            s = torch.atleast_2d(s_in)   #x0 + s.dot(edrs.T)
            vals = 5 * torch.ones(s.size(0)).double()
            x = x0 + s.detach().numpy().dot(edrs)
            x = np.minimum(np.maximum(x,self.lb),self.ub)
            #inbound = np.and(x<self.ub)
            incell = self.kdtree.query(x, eps=0, k=1)[1] == self.target_cell
            vals[incell] = self.GP.LCB_nograd(torch.tensor(x[incell]))
            #print('lcbvals = ',vals)
            return vals
        
        r = r_init
        cyc = 20
        num_sample = num_samples//cyc
        
        s_init = np.zeros(sub_dim)
        #lcb_min = self.GP.LCB_nograd(torch.tensor(x0,device=self.device))
        
        for j in range(cyc):
            search_region = np.array([-r*np.ones(sub_dim),r*np.ones(sub_dim)])
            
            
            s_sample = to_size(latin_hypercube(num_sample, sub_dim), search_region[0], search_region[1])
            lcb_val = 5 * np.ones(num_sample)
            x_sample = np.minimum(np.maximum(x0 + s_sample.dot(edrs),self.lb),self.ub)
            incell = self.kdtree.query(x_sample, eps=0, k=1)[1] == self.target_cell
            
            lcb_val[incell] = self.GP.LCB_nograd(torch.tensor(x_sample[incell],device=self.device)).detach().numpy()
            
            if lcb_val.min() < 0:
                s_init = s_sample[np.argmin(lcb_val)]
                #print('lcb_init = ',lcb_val.min())
                break
            else:
                r = 0.85 * r
        
        sopt, yopt = acq_min_msp(lambda x:lcb(x), lambda x:finite_diff(x,lcb), torch.tensor(s_init,device = self.device), torch.tensor(search_region,device = self.device), n_warmup=10000)
        
        #print('lcb = ',yopt)
        
        x_propose = np.minimum(np.maximum(x0 + sopt.detach().numpy().dot(edrs),self.lb),self.ub)
        #print('lcb_val = ',acq_value)
        return x_propose
    

    def search(self):
        self.random_sample(self.n_init)
        for j in range(self.max_iter-self.n_init):
            print('='*20,'iter',str(j),'='*20)
            self.select_cell()
            
            if not self.use_subspace:
                #x_sample = self.propose_sample_bo(self.num_samples)
                if self.use_approximation:
                    #x_sample = self.propose_sample_LCB_k(self.num_samples)
                    x_sample = self.propose_sample_PI_k(self.num_samples)
                    #x_sample = self.propose_sample_EI_k(self.num_samples)
                else:
                    #x_sample = self.propose_sample_EI_k(self.num_samples)
                    x_sample = self.propose_sample_PI_k(self.num_samples)
                    #x_sample = self.propose_sample_LCB_k(self.num_samples) 
                #x_sample = self.propose_sample_pi(self.num_samples)
            else:
                if j % self.Rp == 0:
                    x_sample = self.propose_sample_subspace_bo(self.num_samples)
                else:
                    x_sample = self.propose_sample_bo(self.num_samples)    
            
            self.collect_sample(x_sample)
            print('current best f(x):',self.y_opt)
            #print('current best x:',self.x_opt)
        
        return (self.x_opt,self.y_opt)
    
    def build_graph(self):
        if self.use_approximation:
            for j in range(self.current_pnts+1):
                ranks = np.argsort(self.dist_matrix[j,:self.current_pnts+1])
                idxes = ranks[:self.N_neighbor]
                
                for idx in idxes:
                    self.adjacent_matrix[j,idx] = self.adjacent_matrix[idx,j] = 1  
        else:
            if not self.target_cell is None:
                self.find_neighbors()
        return 
    
    def find_neighbors(self):
        neighbors = [self.target_cell]
        xs = np.array([s.x for s in self.nodes])
        vor = Voronoi(xs)
        
        for s in vor.ridge_points:
            if self.target_cell in s:
                neighbors.append(s.sum()-self.target_cell)
        
        self.neighbors = np.array(neighbors)
        return
    
    '''
    def build_neighbors(self):
        for j in range(self.current_pnts+1):
            idxes = np.argsort(self.dist_matrix[j,:self.current_pnts+1])[:self.N_neighbor]
            for idx in idxes:
                self.adjacent_matrix[j,idx] = self.adjacent_matrix[idx,j] = 1 
        return 
    '''
    

def orthogonalize(A):                  
    k = A.shape[0]                      
    B=A.copy()                        
    for i in range(1,k):              
        for j in range(i):             
            B[i,:] -= (np.dot(B[j,:],A[i,:])/np.dot(B[j,:],B[j,:]))*B[j,:]
    normB = np.linalg.norm(B, axis = 1)
    C = np.array([B[i,:]/normB[i] for i in range(k) if normB[i] > 1e-6])
    return C

def finite_diff(x_tensor,f,epslong=1e-8):
    with torch.no_grad():
        dims = len(x_tensor)
        delta = epslong*torch.eye(dims,device = x_tensor.device)
        ys = f(torch.cat((x_tensor + delta,x_tensor - delta),dim = 0))
        grad = (ys[:dims] - ys[dims:])/(2*epslong)
    return grad



class node():
    def __init__(self,x,y,idx,n_iter,Cp):
        self.x = x
        self.y = y
        self.c = Cp
        self.Q = -self.y + self.c * sqrt(log(n_iter))
        self.idx = idx
        #self.n_neigh = 0
        
    def get_Q(self,n_iter):
        self.Q = -self.y + self.c * sqrt(log(1+n_iter))  #maximum Q
        return self.Q 


'''
class node():
    def __init__(self,x,y,idx,n_neigh,n_iter,Cp):
        self.x = x
        self.y = y
        self.c = Cp
        self.Q = -self.y + self.c * sqrt(log(n_iter)/n_neigh)
        self.idx = idx
        #self.n_neigh = 0
        
    def update_Q(self,n_neigh,n_iter):
        self.Q = -self.y + self.c * sqrt(log(1+n_iter)/n_neigh)  #maximum Q
        return self.Q 
'''    
