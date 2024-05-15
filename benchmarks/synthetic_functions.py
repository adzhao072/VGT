# -*- coding: utf-8 -*-



import numpy as np
import time
import os

def Levy(x):
    xs = np.atleast_2d(x)*15-5
    dim = xs.shape[1]
    ws = 1 + (xs - 1.0) / 4.0
    val = np.array([np.sin(np.pi * w[0]) ** 2 + \
        np.sum((w[1:dim - 1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[1:dim - 1] + 1) ** 2)) + \
        (w[dim - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[dim - 1])**2) for w in ws])
        
    print('val = ',val)
    with open('result.csv','a+') as f:
        for s in val:
            f.write(str(time.time())+','+str(s)+','+'\n')
    return val

def Michalewicz(xin):
    xs = np.atleast_2d(xin)*np.pi
    dim = xs.shape[1]
    m = 10
    result = -np.sum( np.sin(xs) * np.sin((1+np.arange(dim)) * xs**2 / np.pi) ** (2 * m), axis=-1)

    #result = np.array([100*((x[1:]-x[:-1]**2)**2).sum() + ((x[:-1]-1)**2).sum()  for x in xs])
    print('val = ',result)
    
    with open('result.csv','a+') as f:
        for s in result:
            f.write(str(time.time())+','+str(s)+','+'\n')
    return result


def RosenBrock(xin):
    xs = np.atleast_2d(xin)*4.096-2.048
    result = np.array([100*((x[1:]-x[:-1]**2)**2).sum() + ((x[:-1]-1)**2).sum()  for x in xs])
    print('val = ',result)
    
    with open('result.csv','a+') as f:
        for s in result:
            f.write(str(time.time())+','+str(s)+','+'\n')
            
    
    return result

def Griewank(xin):
    xs = np.atleast_2d(xin)*1000-500
    result = np.array([(x**2).sum()/4000 + 1 - np.cos(x/np.sqrt(range(1,len(x)+1))).prod()  for x in xs])
    with open('result.csv','a+') as f:
        for s in result:
            f.write(str(time.time())+','+str(s)+','+'\n')
    print('val=',result)
    return result



def Schwefel(xin):
    xs = np.atleast_2d(xin)*1000-500
    val = 418.9829 * xs.shape[1] + np.array([(-x*np.sin(np.sqrt(np.abs(x)))).sum() for x in xs])
    with open('result.csv','a+') as f:
        for s in val:
            f.write(str(time.time())+','+str(s)+','+'\n')
    print('val = ',val)
    return val

def Rastrigin(xin):
    xs = np.atleast_2d(xin)*10.24-5.12
    sum_ = np.array([np.dot(x,x)+10*len(x)-10*np.sum(np.cos(2*np.pi*x)) for x in xs])
    with open('result.csv','a+') as f:
        for s in sum_:
            f.write(str(time.time())+','+str(s)+','+'\n')
    # y = np.array([sum_])
    print('yval=',sum_)
    return sum_



def Ackley(xin):
    xs = np.atleast_2d(xin)*15-5
    result = np.array([(-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e ) for x in xs])
    print('val = ',result)
    with open('result.csv','a+') as f:
        for s in result:
            f.write(str(time.time())+','+str(s)+','+'\n')
    return result


def Hartmann6(xin):
    ALPHA = np.array([1.0, 1.2, 3.0, 3.2])

    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
        ])

    P = 0.0001*np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
        ])
    
    xs = np.atleast_2d(xin)
    val = -np.array([(ALPHA*np.exp(-(A*((x-P)**2)).sum(axis=1))).sum() for x in xs])
    print('vals = ',val)
    with open('result.csv','a+') as f:
        for s in val:
            f.write(str(time.time())+','+str(s)+','+'\n')
    return val 



def Hartmann6_500(xin):
    ALPHA = np.array([1.0, 1.2, 3.0, 3.2])

    A = np.array([
        [10, 3, 17, 3.5, 1.7, 8],
        [0.05, 10, 17, 0.1, 8, 14],
        [3, 3.5, 1.7, 10, 17, 8],
        [17, 8, 0.05, 10, 0.1, 14],
        ])

    P = 0.0001*np.array([
        [1312, 1696, 5569, 124, 8283, 5886],
        [2329, 4135, 8307, 3736, 1004, 9991],
        [2348, 1451, 3522, 2883, 3047, 6650],
        [4047, 8828, 8732, 5743, 1091, 381],
        ])
    
    xs1 = np.atleast_2d(xin)
    xs = xs1[:,np.array([3,9,17,23,31,44])]
    val = -np.array([(ALPHA*np.exp(-(A*((x-P)**2)).sum(axis=1))).sum() for x in xs])
    print('vals = ',val)
    with open('result.csv','a+') as f:
        for s in val:
            f.write(str(time.time())+','+str(s)+','+'\n')
    return val 



def Ackley10_500(xin):
    xs1 = np.atleast_2d(xin)*15-5
    xs = xs1[:,np.array([3,9,17,23,31,44,233,157,324,412])]
    result = np.array([(-20*np.exp(-0.2 * np.sqrt(np.inner(x,x) / x.size )) -np.exp(np.cos(2*np.pi*x).sum() /x.size) + 20 +np.e ) for x in xs])
    print('val = ',result)
    with open('result.csv','a+') as f:
        for s in result:
            f.write(str(time.time())+','+str(s)+','+'\n')
    return result








