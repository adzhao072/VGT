# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:36:09 2023

@author: ZhaoAidong
"""

import torch
import numpy as np
import matplotlib.pyplot as plt



class linear_regression():
    def __init__(self,X,y,w0=None):
        self.X=X
        self.y=y
        self.dims = X.shape[1]
        self.npnts = X.shape[0]
        if w0 is None:
            self.w = torch.ones(self.dims, device = X.device, dtype=X.dtype, requires_grad=True)
        else:
            self.w = w0
        
    def forward(self):
        return torch.mv(self.X,self.w)
    
    def predict(self,x):
        return torch.mv(x,self.w).clone().detach()
    
    def loss(self):
        return torch.sum((self.forward() - self.y) ** 2) + 0.1*(self.w**2).sum()/self.dims*self.npnts
    
    
def train_lin_reg(X,y,num_step=200):
    model = linear_regression(X,y)
    optimizer = torch.optim.Adam([{"params": model.w}], lr=0.1)
    loss_list = []
    for i in range(num_step):
        optimizer.zero_grad()
        # making predictions with forward pass
        model.forward()
        # calculating the loss between original and predicted data points
        loss = model.loss()
        # storing the calculated loss in a list
        loss_list.append(loss.item())
        # backward pass for computing the gradients of the loss w.r.t to learnable parameters
        loss.backward()
        optimizer.step()
        # priting the values for understanding
        #print('{},\t{}'.format(i, loss.item()))
    
    return model,loss_list



if __name__=="__main__":
    pnts=200
    dims = 1000
    
    X = torch.randn(pnts,dims)
    #X = torch.arange(-5, 5, 0.1).view(-1, 1)
    func = (3 * X).sum(dim=1)
    Y = func + 0.1 * torch.randn(pnts)
    model,loss_list = train_lin_reg(X, Y,num_step=200)
    
    print(((model.predict(X)-Y)**2).sum().item())
    # Plotting the loss after each iteration
    plt.plot(loss_list, 'r')
    plt.tight_layout()
    plt.grid('True', color='y')
    plt.xlabel("Epochs/Iterations")
    plt.ylabel("Loss")
    plt.show()