import torch


def latin_hypercube(n, dims, device):
    points = torch.zeros((n, dims),device = device)
    centers = torch.arange(0.0, n, device = device)
    centers = centers / float(n)
    for i in range(0, dims):
        points[:, i] = centers[torch.randperm(n,device = device)]

    perturbation = torch.rand(n, dims, device = device) 
    perturbation = perturbation / float(n)
    points += perturbation
    return points

def is_legal(v):
    legal = not torch.isnan(v).any() and not torch.isinf(v).any()
    return legal


class lbfgsb():
    def __init__(self, f, f_grad, xl, xu, max_iter = 20, lr=1., history_size=10, dtype=torch.double, device = 'cpu'):
        self.lr = lr
        self.f = f
        self.f_grad = f_grad
        self.xl = xl.to(device).double()
        self.xu = xu.to(device).double()
        self.device = device
        self.dtype = dtype
        self.history_size = history_size
        self.max_iter = max_iter

    def two_loop_recursion(self, vec):
        num_old = len(self.old_dirs)

        rho = [1. / self.old_stps[i].dot(self.old_dirs[i]) for i in range(num_old)]
        alpha = []
        q = vec
        
        for i in reversed(range(num_old)):
            #print('q=',q,'self.old_dirs[i] = ',self.old_dirs[i])
            tmp_alpha = self.old_dirs[i].dot(q) * rho[i]
            alpha.append(tmp_alpha)
            q.add_(tmp_alpha*self.old_stps[i])

        r = torch.mul(q, self.H_diag)
        for i in range(num_old):
            beta = self.old_stps[i].dot(r) * rho[i]
            r.add_((alpha[i] - beta)*self.old_dirs[i])

        return r

    def curvature_update(self, flat_grad, eps=1e-2, damping=False):
        # compute y's
        y = flat_grad.sub(self.prev_flat_grad)
        sBs = self.s.dot(self.Bs)
        
        #print('y = ',y,'s = ',self.s)
        ys = y.dot(self.s)  # y*s

        # update L-BFGS matrix
        if ys > eps * sBs or damping == True:

            # perform Powell damping
            if damping == True and ys < eps*sBs:
                theta = ((1 - eps) * sBs)/(sBs - ys)
                y = theta * y + (1 - theta) * self.Bs

            # updating memory
            if len(self.old_dirs) == self.history_size:
                # shift history by one (limited-memory)
                self.old_dirs.pop(0)
                self.old_stps.pop(0)

            # store new direction/step
            self.old_dirs.append(self.s)
            self.old_stps.append(y)

            # update scale of initial Hessian approximation
            self.H_diag = ys / y.dot(y)  # (y*y)
        else:
            # save skip
            self.curv_skips += 1

        return

    def step(self, p_k, g_Ok, g_Sk=None):
        # keep track of nb of iterations
        self.n_iter += 1

        # modify previous gradient
        if self.prev_flat_grad is None:
            self.prev_flat_grad = g_Ok.clone()
        else:
            self.prev_flat_grad.copy_(g_Ok)

        # set initial step size: self.t

        if g_Sk is None:
            g_Sk = g_Ok.clone()

        # perform update
        s = self.t * p_k 
        
        self.s = torch.maximum(torch.minimum(s,self.xu - self.x), self.xl-self.x)#.float()

        # store Bs
        if self.Bs is None:
            self.Bs = (g_Sk.mul(-self.s)).clone()
        else:
            self.Bs.copy_(g_Sk.mul(-self.s))

        return self.s
    
    def optimize(self,x0, eps = 1e-2, damping = False):
        self.n_iter = 0
        self.curv_skips = 0
        self.H_diag = 1

        self.old_dirs = []
        self.old_stps = []
        self.Bs = None
        self.prev_flat_grad = None
        
        self.x = x0.to(self.device).double()
        new_f = self.f(self.x).item()#.squeeze()#
        grad_f = self.f_grad(self.x)#.squeeze()#.double()
        tmplrs = 0.5**torch.arange(20,device = self.device).double()
        f_vals = self.f(torch.maximum(torch.minimum(self.x - tmplrs.reshape(-1,1).mm(grad_f.reshape(1,-1)),self.xu), self.xl))
        idx = torch.nonzero(f_vals<new_f,as_tuple=False).squeeze()
        #if torch.any(f_vals<new_f).item():
        if len(idx.shape)>0 and idx.shape != torch.Size([0]):
            #print('idx = ',idx[0])
            self.t = tmplrs[idx[0]]/2
            for k in range(self.max_iter):
                if self.n_iter > 0:
                    self.curvature_update(grad_f, eps, damping)
                
                # compute search direction
                p = self.two_loop_recursion(-grad_f)
                
                # take step
                s = self.step(p, grad_f)
                f_next = self.f(self.x.add_(s)).item()
                
                if f_next < new_f:
                    new_f = f_next
                    self.x.add_(s)
                    grad_f = self.f_grad(self.x)#.squeeze()
                else:
                    break
                
                l_inf_norm_grad = torch.max(torch.abs(grad_f)).item()
                l_inf_norm_s = torch.max(torch.abs(self.s)).item()
                
                if l_inf_norm_grad<1e-6 or l_inf_norm_s<1e-6 or not is_legal(grad_f):
                    break

        return self.x,new_f
        
def acq_min_msp(f, gradf, x_init, bounds, n_warmup=10000, n_iter=1):
    device = bounds.device
    dims = bounds.size(1)
    rand_points = latin_hypercube(n_warmup, dims, device)
    x_tries = bounds[0] + (bounds[1]-bounds[0]) * rand_points
    ys = f(x_tries)
    idx = torch.argsort(ys)
    min_acq = ys[idx[0]]
    x_min = x_tries[idx[0]]
    
    #print('min_acq :',min_acq)
    if f(x_init)<min_acq:
        x_min = x_init
        min_acq = f(x_init)
        #print('min_acq :',min_acq)
    
    for k in range(n_iter):
        optimizer = lbfgsb(f, gradf, bounds[0],bounds[1], device=device)
        xopt, yopt = optimizer.optimize(x_min)
        if min_acq is None or yopt <= min_acq:
            x_min = xopt
            min_acq = yopt
    return x_min, min_acq
