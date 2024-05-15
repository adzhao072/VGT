


import numpy as np
from numpy import linalg

class SIR:
    def __init__(self, K = 2, H = 5, bins = None):
        self.K = K
        self.H = H
        #default is equally spaced bins
        self.bins = bins

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

        #n is the number of observations
        n = X.shape[0]
        p = X.shape[1]

        x_bar = np.mean(X,axis =0)


        #compute the bins, assuming default 
        if self.bins == None:
        	n_h, bins = np.histogram(Y,bins = min(self.H,n//3))  # parameter H
        else: 
        	n_h,bins = np.histogram(Y, bins = self.bins)

        #assign a bin to each observations
        assignments = np.digitize(Y,bins)

        #this is really hacky... 
        assignments[np.argmax(assignments)] -= 1

        #loop through the slices, for each slice compute within slice mean
        M = np.zeros((p,p))
        for i in range(len(n_h)):

        	h = n_h[i]
        	if h != 0:
        		x_h_bar = np.mean(X[assignments == i + 1],axis = 0)
        	else:
        		x_h_bar = np.zeros(p)

        	x_std = x_h_bar - x_bar

        	M += float(h) * np.outer(x_std,x_std)

        #compute the estimate of the covariance matrix M
        self.M = M/n

        #eigendecomposition of V
        cov = np.cov(X.T)
        V = np.dot(linalg.inv(cov),M)
        eigenvalues, eigenvectors = linalg.eig(V)

        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:,idx]

        #assign first K columns to beta 
        beta = eigenvectors[:,0:self.K]
        self.beta = beta
        self.eigenvalues = eigenvalues
        return beta

    def transform(self, X_to_predict):
    	beta = self.beta 
    	return np.dot(X_to_predict,beta)



def test_fun(x):
    return x.sum()#np.dot(x,x)

if __name__ == '__main__':
    x=np.random.rand(10,20)*2
    y=np.array([test_fun(s) for s in x])
    
    
    print(0.9*np.ones(20))
    
    
    sir = SIR()
    sir.fit(x,y)
    #print(sir.beta)
    #print(sir.eigenvalues)
    y_pre = sir.transform(0.5*np.ones(20))
    print('festures = ',y_pre)
    print(np.dot(y_pre,y_pre))

    
