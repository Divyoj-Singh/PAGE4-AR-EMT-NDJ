# -*- coding: utf-8 -*-
"""
Created on Thu Nov  19  2020

@author: divyoj
"""
#  Auxilary functions.
# This file has the necessary functions to draw bifurcation diagram for EMT-AR-Page-4-NDJ network. Its in Python3.
# it also has fuctions to draw dynamic behaviour. 
# it is cpu-parallelised.
############################################
#%% importing packages:
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing as mp
import random
import seaborn as sns
import pandas as pd
import time

#%% Control parametes:
# Specify the number of cpu-cores to be used: 
no_of_cpu_for_use=35
step_size_for_storing=1000; ## in dynamic_behaviour
# specify the number of initial conditions to start with, for finding the steady states
no_of_initial_conditions_for_bifurcation=70
#----------------------------------------------------------------------------------------#
# parameters used in the simulations
def parameters(dt):
    dic =  {#----------EMT circuit----------
            # production rates:
           'gu200' : 2.1e+3,  'gZ': 1.1e+1
            # hill coefficient:
           ,'nZu200': 3.0e+0,  'nu200' : 6.0e+0, 'nSu200': 2.0e+0, 'nSZ': 2.0e+0, 'nZZ': 2.0e+0
            # lambda values(interaction strength):
           ,'lZu200': 1.0e-1,  'lSu200': 1.0e-1, 'lSZ'   : 1.0e+1,'lZZ': 7.5e+0
            # degradation rates:
           ,'gammau200' : 5.0e-2,  'gammaZ' : 1.0e-1
            # hill function threshold value:
           ,'S0u200': 1.8e+5,  'S0Z': 1.8e+5,'Z0u200': 2.2e+5, 'Z0Z': 2.5e+4
            ## mu34 interactions:
           ,'nSu34' : 1.0e+0, 'nSS': 1.0e+0, 'nu34' : 2.0e+0, 'nI' : 2.0e+0   
           ,'lSu34' : 1.0e-1, 'lSS': 1.0e-1, 'lZu34': 2.0e-1, 'lIS': 6.5e+0
           ,'gammau34'  : 5.0e-2, 'gammaS' : 1.25e-1
           ,'gu34'  : 1.35e+3,'gS' : 9.0e+1
           ,'S0u34' : 3.0e+5, 'S0S': 2.0e+5
           ,'Z0u34' : 6.0e+5
           ,'u034'  : 1.0e+4
           ,'I0S'   : 3.0e+2
            ## micro-array interaction:
           ,'u0200' : 1.0e+4
           ,'gu':[0.0e+0, 1*5.0e-3, 2*5.0e-2, 3*5.0e-1, 4*5.0e-1, 5*5.0e-1, 6*5.0e-1]
           ,'gm':[0.0e+0, 4.0e-2, 2.0e-1, 1.0e+0, 1.0e+0, 1.0e+0, 1.0e+0]
           ,'l':[1.0e+0, 6.0e-1, 3.0e-1, 1.0e-1, 5.0e-2, 5.0e-2, 5.0e-2]
           ,'km' : 5.0e-1,'kP' : 1.0e+2
           #-------------------------------------------
           #----------Notch Signaling circuit----------
           ,'gammaN'  : 1.0e-1, 'gammaI' : 5.0e-1                                    
           ,'kc' : 1.0e-4, 'kt' : 1.0e-5                                   
           ,'p'  : 2.0e+0, 'pf' : 1.0            
           ,'gN' : 0.8e+1, 'gD' : 7.0e+1, 'gJ' : 2.0e+1     
           ,'I0' : 1.0e+2                                   
           ,'Nt' : 0.1e+0, 'Dt' : 0.1e+0, 'Jt' : 0.1e+0 
           ,'Nn' : 0.1e+0, 'Dn' : 0.1e+0, 'Jn' : 0.1e+0 
           ,'ln' : 7.0e+0, 'ld' : 0.0e+0, 'lj' : 2.0e+0   
           ,'ldf': 3.0,    'ljf': 0.3
           ,'It' : 0.0            
           #----------Page4-AR circuit----------
           ,'H': 2.31  
           ,'gA' : 4.62,'gP' : 2.31,'gC':2.77,'gH':0.04
            #-----------------------------------
           ,'gammaC':0.02,'gammaPM':0.004,'gammaPH': 0.04,'gammaPU': 0.016 ,'gammaA':0.020
            #------------------------
           ,'a':5,'b':20
           ,'tau_A':16.23/dt,'tau_C':16.23/dt ## delay term (divided by dt to convert it into steps.)
           ,'nA': 4,'nC':4
           ,'P0_A':20,'A0':65
           
           
           ## lambdas of hill functions coupling AR with the Page4 circuit##
           ,'lMtoA':0.9 # H-P4 to AR
           ,'lAtoC':0.9 # AR to CLK2
            
            # parameters for duble negative feedback with ZEB:
            ,'nZtoA':4,'nAtoZ':4
            ,'lAtoZ': 0.9 # AR to ZEB
            ,'lZtoA':0.9 # ZEB to AR
            ,'Z0A': 2.5e+4 ## same as Z to Z
            ,'A0Z':65 # same as threshold for other interations for AR
            ## initial conditions:            
            ,'Zic':40000.0,'Sic':200000.0,'Wic':20000.0,'Yic':20000.0
            ,'Nic':0.0e+0,'Dic':0.0e+0,'Jic':0.0e+0,'Iic':20.0e+0
            }
    return dic

def step(x,tau):
    # define step function for delay term
    return 1 * (x > tau)

def HS(X,X0,nX,lamb):
    # Shifted hill function for the interactions:
    return lamb + (1.0-lamb)/(1.0 + (X/X0)**nX)

def M(X,X0,i,n):
    # function M for the micro rna interaction
    return ((X/X0)**i)/((1. + (X/X0))**n)

def C(i,n):
    # function C for the micro rna interaction (uses the function gamma from scipy)
    return gamma(n+1)/(gamma(n-i+1)*gamma(i+1))

def Py(X, k, n,u0, gu,gm):
    # function Py for the micro rna interaction
	v1 = 0
	v2 = 0
	for i in range(n+1):
		v1 += gu[i]*C(i,n)*M(X,u0,i,n)
		v2 += gm[i]*C(i,n)*M(X,u0,i,n)
	return v1/(v2+k)

def Pl(X, k, n,u0 ,l,gm):
    # function Pl for the micro rna interaction
	v1 = 0
	v2 = 0
	for i in range(n+1):
		v1 +=  l[i]*C(i,n)*M(X,u0,i,n)
		v2 += gm[i]*C(i,n)*M(X,u0,i,n)
	return v1/(v2+k)

def shorten_the_array(array,no_of_steps_to_skip):
    array_to_return=np.empty([]);
    for i in range(0,array.size):
        if (i%step_size_for_storing)==0:
            array_to_return=np.append(array_to_return,array[i])
    array_to_return=np.delete(array_to_return,0)
    return array_to_return 

#%% Euler Integration of ODES:
def integrate_cons(k,i,time, p):
    # function to intergrate the differential equations over time.
    dt = time[1]-time[0]  #0.1
    npoints = time.size - 1 #int(T/dt)

    # define vectors to store results
    PU = np.empty(npoints+1);PM = np.empty(npoints+1);
    PH = np.empty(npoints+1);C = np.empty(npoints+1);A = np.empty(npoints+1);
    Z = np.empty(npoints+1);W = np.empty(npoints+1);S = np.empty(npoints+1);Y = np.empty(npoints+1);
    N = np.empty(npoints+1);D = np.empty(npoints+1);J = np.empty(npoints+1);I = np.empty(npoints+1);
    
    # setting the seed value for random number generation:
    random.seed(i)
    # setting the initial conditions:
    PM[0]=0.;PH[0]=0.;C[0]=0.;PU[0]=0.;
    W[0]=random.randint(0,50000);    Z[0]=random.randint(0,5000000);Y[0]=random.randint(0,50000);    S[0]=random.randint(0,500000); 
    
    A[0]=random.randint(0,300);
    
    N[0]=random.randint(0,500000); D[0]=random.randint(0,500000);J[0]=random.randint(0,500000);I[0]=random.randint(0,500000);
    
    
    # integrating over time:
    for i in range(1,npoints+1):
        PU[i] = PU[i-1] + dt*( p['gP'] - p['H']*PU[i-1]/(PU[i-1]+p['a']) - p['gammaPU']*PU[i-1] ) # WT-page4
        PM[i] = PM[i-1] + dt*( p['H']*PU[i-1]/(PU[i-1]+p['a']) - p['gH']*C[i-1]*PM[i-1]/(PM[i-1]+p['b']) - p['gammaPM']*PM[i-1] )#HIPK1-PAGE4
        PH[i] = PH[i-1] + dt*( p['gH']*C[i-1]*PM[i-1]/(PM[i-1]+p['b']) - p['gammaPH']*PH[i-1] )#CLK2-PAGE4 (PH)
        C[i] = C[i-1] + dt*( p['gC']*HS(A[int(i-1-p['tau_C']*step(i,p['tau_C']))],p['A0'],p['nC'],p['lAtoC'])-p['gammaC']*C[i-1])# CLK2
 
        A[i] = A[i-1] + dt*( p['gA']*HS(Z[i-1],p['Z0A'],p['nZtoA'],p['lZtoA'])*HS(PM[int(i-1-p['tau_A']*step(i,p['tau_A']))],p['P0_A'],p['nA'],p['lMtoA']) - p['gammaA']*A[i-1] ) # Androgen
        Z[i] = Z[i-1] + dt*( p['kP']*p['gZ']*HS(A[i-1],p['A0Z'],p['nAtoZ'],p['lAtoZ'])*HS(Z[i-1],p['Z0Z'],p['nZZ'],p['lZZ'])*HS(S[i-1],p['S0Z'],p['nSZ'],p['lSZ'])*Pl(W[i-1],p['km'],6,p['u0200'],p['l'],p['gm']) - p['gammaZ']*Z[i-1] ) #Zeb
        W[i] = W[i-1] + dt*(p['gu200']*HS(Z[i-1],p['Z0u200'],p['nZu200'],p['lZu200'])*HS(S[i-1],p['S0u200'],p['nSu200'],p['lSu200']) - p['gZ']*HS(Z[i-1],p['Z0Z'],p['nZZ'],p['lZZ'])*HS(S[i-1],p['S0Z'],p['nSZ'],p['lSZ'])*Py(W[i-1],p['km'],6,p['u0200'],p['gu'],p['gm'])-p['gJ']*HS(I[i-1],p['I0'],p['p'],p['lj'])*Py(W[i-1],p['km'],5,p['u0200'],p['gu'],p['gm']) - p['gammau200']*W[i-1]) ## mir-200 
        S[i] = S[i-1] + dt*(p['kP']*p['gS']*HS(S[i-1],p['S0S'],p['nSS'],p['lSS'])*HS(I[i-1],p['I0S'],p['nI'],p['lIS'])*HS(p['It'],p['I0S'],p['nI'],p['lIS'])*Pl(Y[i-1],p['km'],2,p['u034'],p['l'],p['gm'])  - p['gammaS']*S[i-1])
        
        Y[i] = Y[i-1] + dt*(p['gu34']*HS(S[i-1],p['S0u34'],p['nSu34'],p['lSu34'])*HS(Z[i-1],p['Z0u34'],p['nu34'],p['lZu34']) - p['gS']*HS(S[i-1],p['S0S'],p['nSS'],p['lSS'])*HS(I[i-1],p['I0S'],p['nI'],p['lIS'])*HS(p['It'],p['I0S'],p['nI'],p['lIS'])*Py(Y[i-1],p['km'],2,p['u034'],p['gu'],p['gm']) - p['gN']*HS(I[i-1],p['I0'],p['p'],p['ln'])*Py(Y[i-1],p['km'],2,p['u034'],p['gu'],p['gm']) - p['gD']*HS(I[i-1],p['I0'],p['p'],p['ld'])*Py(Y[i-1],p['km'],3,p['u034'],p['gu'],p['gm']) - p['gammau34']*Y[i-1])
        
        N[i] = N[i-1] + dt*(p['kP']*p['gN']*HS(I[i-1],p['I0'],p['p'],p['ln'])*Pl(Y[i-1],p['km'],2,p['u034'],p['l'],p['gm']) - N[i-1]*( (p['kc']*D[i-1] + p['kt']*p['Dt'])*HS(I[i-1],p['I0'],p['pf'],p['ldf']) + (p['kc']*J[i-1] + p['kt']*p['Jt'])*HS(I[i-1],p['I0'],p['pf'],p['ljf']) ) - p['gammaN']*N[i-1])
        D[i] = D[i-1] + dt*(p['kP']*p['gD']*HS(I[i-1],p['I0'],p['p'],p['ld'])*Pl(Y[i-1],p['km'],3,p['u034'],p['l'],p['gm']) - D[i-1]*(p['kc']*N[i-1]*HS(I[i-1],p['I0'],p['pf'],p['ldf']) + p['kt']*p['Nt'] ) - p['gammaN']*D[i-1])
        J[i] = J[i-1] + dt*(p['kP']*p['gJ']*HS(I[i-1],p['I0'],p['p'],p['lj'])*Pl(W[i-1],p['km'],5,p['u0200'],p['l'],p['gm']) - J[i-1]*(p['kc']*N[i-1]*HS(I[i-1],p['I0'],p['pf'],p['ljf']) + p['kt']*p['Nt'] ) - p['gammaN']*J[i-1])
        I[i] = I[i-1] + dt*(p['kt']*N[i-1]*(p['Dt']*HS(I[i-1],p['I0'],p['pf'],p['ldf']) + p['Jt']*HS(I[i-1],p['I0'],p['pf'],p['ljf']) ) - p['gammaI']*I[i-1])
    
    A=shorten_the_array(A,step_size_for_storing);             
    Z=shorten_the_array(Z,step_size_for_storing);             
    S=shorten_the_array(S,step_size_for_storing);             
    W=shorten_the_array(W,step_size_for_storing);             
    Y=shorten_the_array(Y,step_size_for_storing);             
    N=shorten_the_array(N,step_size_for_storing);             
    D=shorten_the_array(D,step_size_for_storing);             
    J=shorten_the_array(J,step_size_for_storing);             
    I=shorten_the_array(I,step_size_for_storing);             
    return k,A,Z,S,W,Y,N,D,J,I

def steady_state(k,i,time, p):
    # function to intergrate the differential equations over time and return the steady state:

    dt = time[1]-time[0]  #0.1
    npoints = time.size - 1 #int(T/dt)

    # define vectors to store results
    PU = np.empty(npoints+1);PM = np.empty(npoints+1);
    PH = np.empty(npoints+1);C = np.empty(npoints+1);A = np.empty(npoints+1);
    Z = np.empty(npoints+1);W = np.empty(npoints+1);S = np.empty(npoints+1);Y = np.empty(npoints+1);
    N = np.empty(npoints+1);D = np.empty(npoints+1);J = np.empty(npoints+1);I = np.empty(npoints+1);
    
    # setting the seed value for random number generation:
    random.seed(i)
    # setting the initial conditions:
    PM[0]=0.;PH[0]=0.;C[0]=0.;PU[0]=0.;W[0]=random.randint(0,50000);Z[0]=random.randint(0,5000000);
    Y[0]=random.randint(0,50000);S[0]=random.randint(0,500000); 
    
    A[0]=random.randint(0,300);N[0]=random.randint(0,500000);D[0]=random.randint(0,500000);
    J[0]=random.randint(0,500000);I[0]=random.randint(0,500000);
    
    # integrating over time:
    for i in range(1,npoints+1):
        PU[i] = PU[i-1] + dt*( p['gP'] - p['H']*PU[i-1]/(PU[i-1]+p['a']) - p['gammaPU']*PU[i-1] ) # WT-page4
        PM[i] = PM[i-1] + dt*( p['H']*PU[i-1]/(PU[i-1]+p['a']) - p['gH']*C[i-1]*PM[i-1]/(PM[i-1]+p['b']) - p['gammaPM']*PM[i-1] )#HIPK1-PAGE4
        PH[i] = PH[i-1] + dt*( p['gH']*C[i-1]*PM[i-1]/(PM[i-1]+p['b']) - p['gammaPH']*PH[i-1] )#CLK2-PAGE4 (PH)
        C[i] = C[i-1] + dt*( p['gC']*HS(A[int(i-1-p['tau_C']*step(i,p['tau_C']))],p['A0'],p['nC'],p['lAtoC'])-p['gammaC']*C[i-1])# CLK2
 
        A[i] = A[i-1] + dt*( p['gA']*HS(Z[i-1],p['Z0A'],p['nZtoA'],p['lZtoA'])*HS(PM[int(i-1-p['tau_A']*step(i,p['tau_A']))],p['P0_A'],p['nA'],p['lMtoA']) - p['gammaA']*A[i-1] ) # Androgen
        Z[i] = Z[i-1] + dt*( p['kP']*p['gZ']*HS(A[i-1],p['A0Z'],p['nAtoZ'],p['lAtoZ'])*HS(Z[i-1],p['Z0Z'],p['nZZ'],p['lZZ'])*HS(S[i-1],p['S0Z'],p['nSZ'],p['lSZ'])*Pl(W[i-1],p['km'],6,p['u0200'],p['l'],p['gm']) - p['gammaZ']*Z[i-1] ) #Zeb
        W[i] = W[i-1] + dt*(p['gu200']*HS(Z[i-1],p['Z0u200'],p['nZu200'],p['lZu200'])*HS(S[i-1],p['S0u200'],p['nSu200'],p['lSu200']) - p['gZ']*HS(Z[i-1],p['Z0Z'],p['nZZ'],p['lZZ'])*HS(S[i-1],p['S0Z'],p['nSZ'],p['lSZ'])*Py(W[i-1],p['km'],6,p['u0200'],p['gu'],p['gm'])-p['gJ']*HS(I[i-1],p['I0'],p['p'],p['lj'])*Py(W[i-1],p['km'],5,p['u0200'],p['gu'],p['gm']) - p['gammau200']*W[i-1]) ## mir-200 
        S[i] = S[i-1] + dt*(p['kP']*p['gS']*HS(S[i-1],p['S0S'],p['nSS'],p['lSS'])*HS(I[i-1],p['I0S'],p['nI'],p['lIS'])*HS(p['It'],p['I0S'],p['nI'],p['lIS'])*Pl(Y[i-1],p['km'],2,p['u034'],p['l'],p['gm'])  - p['gammaS']*S[i-1])
        
        Y[i] = Y[i-1] + dt*(p['gu34']*HS(S[i-1],p['S0u34'],p['nSu34'],p['lSu34'])*HS(Z[i-1],p['Z0u34'],p['nu34'],p['lZu34']) - p['gS']*HS(S[i-1],p['S0S'],p['nSS'],p['lSS'])*HS(I[i-1],p['I0S'],p['nI'],p['lIS'])*HS(p['It'],p['I0S'],p['nI'],p['lIS'])*Py(Y[i-1],p['km'],2,p['u034'],p['gu'],p['gm']) - p['gN']*HS(I[i-1],p['I0'],p['p'],p['ln'])*Py(Y[i-1],p['km'],2,p['u034'],p['gu'],p['gm']) - p['gD']*HS(I[i-1],p['I0'],p['p'],p['ld'])*Py(Y[i-1],p['km'],3,p['u034'],p['gu'],p['gm']) - p['gammau34']*Y[i-1])
        
        N[i] = N[i-1] + dt*(p['kP']*p['gN']*HS(I[i-1],p['I0'],p['p'],p['ln'])*Pl(Y[i-1],p['km'],2,p['u034'],p['l'],p['gm']) - N[i-1]*( (p['kc']*D[i-1] + p['kt']*p['Dt'])*HS(I[i-1],p['I0'],p['pf'],p['ldf']) + (p['kc']*J[i-1] + p['kt']*p['Jt'])*HS(I[i-1],p['I0'],p['pf'],p['ljf']) ) - p['gammaN']*N[i-1])
        D[i] = D[i-1] + dt*(p['kP']*p['gD']*HS(I[i-1],p['I0'],p['p'],p['ld'])*Pl(Y[i-1],p['km'],3,p['u034'],p['l'],p['gm']) - D[i-1]*(p['kc']*N[i-1]*HS(I[i-1],p['I0'],p['pf'],p['ldf']) + p['kt']*p['Nt'] ) - p['gammaN']*D[i-1])
        J[i] = J[i-1] + dt*(p['kP']*p['gJ']*HS(I[i-1],p['I0'],p['p'],p['lj'])*Pl(W[i-1],p['km'],5,p['u0200'],p['l'],p['gm']) - J[i-1]*(p['kc']*N[i-1]*HS(I[i-1],p['I0'],p['pf'],p['ljf']) + p['kt']*p['Nt'] ) - p['gammaN']*J[i-1])
        I[i] = I[i-1] + dt*(p['kt']*N[i-1]*(p['Dt']*HS(I[i-1],p['I0'],p['pf'],p['ldf']) + p['Jt']*HS(I[i-1],p['I0'],p['pf'],p['ljf']) ) - p['gammaI']*I[i-1])
       
    return k,A[-1],Z[-1],S[-1],W[-1],Y[-1],N[-1],D[-1],J[-1],I[-1]

 
def different_states(time,p):
    AR=[];zeb=[];snail=[];mir200=[];mir34=[];notch=[];delta=[];jagged=[];nicd=[];
    index_array = np.arange(1,no_of_initial_conditions_for_bifurcation,1)
    temp_AR=[];temp_zeb=[];temp_snail=[];temp_mir200=[];temp_mir34=[];temp_notch=[];temp_delta=[];temp_jagged=[];temp_nicd=[];

    pool = mp.Pool(no_of_cpu_for_use)
    output= pool.starmap_async(steady_state, [(k,i,time, p) for k, i in enumerate(index_array)]).get()
    
    for l in output:
        temp_AR.append(l[1])
        temp_zeb.append(l[2])
        temp_snail.append(l[3])
        temp_mir200.append(l[4])
        temp_mir34.append(l[5])
        temp_notch.append(l[6])
        temp_delta.append(l[7])
        temp_jagged.append(l[8])
        temp_nicd.append(l[9])
                
        pool.close()
     
    temp_AR=np.around(temp_AR,0);AR=list(set(temp_AR))
    temp_zeb=np.around(temp_zeb,-1);zeb=list(set(temp_zeb));temp_snail=np.around(temp_snail,-1);snail=list(set(temp_snail))
    temp_mir200=np.around(temp_mir200,-1);mir200=list(set(temp_mir200));temp_mir34=np.around(temp_mir34,-1);mir34=list(set(temp_mir34))
    
    temp_nicd=np.around(temp_nicd,0);nicd=list(set(temp_nicd));temp_notch=np.around(temp_notch,0);notch=list(set(temp_notch))
    temp_delta=np.around(temp_delta,0);delta=list(set(temp_delta));temp_jagged=np.around(temp_jagged,0);jagged=list(set(temp_jagged))
    
    return AR,zeb,snail,mir200,mir34,notch,delta,jagged,nicd

def numerical_bifurcation(bifur_para,bifur_para_array,tm,p):
    t=time.time()
    bifur_dict={}
    for i in bifur_para_array:
        p[bifur_para]=i
        #print(p['It'])
        bifur_dict[i]={"AR":[],"zeb":[],"snail":[],"mir200":[],"mir34":[],"notch":[],"delta":[],"jagged":[],"nicd":[]}
        bifur_dict[i]["AR"],bifur_dict[i]["zeb"],bifur_dict[i]["snail"],bifur_dict[i]["mir200"],bifur_dict[i]["mir34"],bifur_dict[i]["notch"],bifur_dict[i]["delta"],bifur_dict[i]["jagged"],bifur_dict[i]["nicd"]=  different_states(tm, p)

    ## Saving data:
    csv_filename = "lAtoZ=" + str(p["lAtoZ"]) + "_lZtoA=" + str(p["lZtoA"]) + "_gD=" + str(p["gD"])+ "_gJ=" + str(p["gJ"]) + "_" + bifur_para + "=" + str(bifur_para_array[0]) + "_" + str(bifur_para_array[-1])
    np.save(csv_filename,bifur_dict)
    dff=pd.DataFrame.from_dict(bifur_dict,orient='index')
    dff.to_csv(csv_filename + ".csv")
    
    ### plotting:    
    f, axs = plt.subplots(3,3,figsize=(15,7.5)) 
    for key in bifur_dict:
        sns.scatterplot(x=key,y=bifur_dict[key]["AR"],ax=axs[0,0])
        axs[0,0].set_ylabel("AR values")
        axs[0,0].set_xlabel("bifurcation paramter "+bifur_para)

        sns.scatterplot(x=key,y=bifur_dict[key]["zeb"],ax=axs[0,1])
        axs[0,1].set_ylabel("zeb values")
        axs[0,1].set_xlabel("bifurcation paramter "+bifur_para)

        sns.scatterplot(x=key,y=bifur_dict[key]["snail"],ax=axs[0,2])
        axs[0,2].set_ylabel("snail values")
        axs[0,2].set_xlabel("bifurcation paramter "+bifur_para)
        
        sns.scatterplot(x=key,y=bifur_dict[key]["mir200"],ax=axs[1,0])
        axs[1,0].set_ylabel("mir200 values")
        axs[1,0].set_xlabel("bifurcation paramter "+bifur_para)

        sns.scatterplot(x=key,y=bifur_dict[key]["mir34"],ax=axs[1,1])
        axs[1,1].set_ylabel("mir34 values")
        axs[1,1].set_xlabel("bifurcation paramter "+bifur_para)
        
        sns.scatterplot(x=key,y=bifur_dict[key]["notch"],ax=axs[1,2])
        axs[1,2].set_ylabel("notch values")
        axs[1,2].set_xlabel("bifurcation paramter "+bifur_para)

        sns.scatterplot(x=key,y=bifur_dict[key]["delta"],ax=axs[2,0])
        axs[2,0].set_ylabel("delta values")
        axs[2,0].set_xlabel("bifurcation paramter "+bifur_para)

        sns.scatterplot(x=key,y=bifur_dict[key]["jagged"],ax=axs[2,1])
        axs[2,1].set_ylabel("jagged values")
        axs[2,1].set_xlabel("bifurcation paramter "+bifur_para)

        sns.scatterplot(x=key,y=bifur_dict[key]["nicd"],ax=axs[2,2])
        axs[2,2].set_ylabel("nicd values")
        axs[2,2].set_xlabel("bifurcation paramter "+bifur_para)

    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.5)
    f.suptitle(csv_filename)
    plt.savefig(csv_filename+".png",dpi=200)
    plt.show()
    print("Simulation time(hrs):",(time.time()-t)/3600)
    
def multiple_dynamic(tm,p,n,folder):
    t=time.time()
    plot_dict={}
    index_array=np.arange(0,n,1)	
    for i in index_array:
        plot_dict[i]={"AR":[],"zeb":[],"snail":[],"mir200":[],"mir34":[],"notch":[],"delta":[],"jagged":[],"nicd":[]}
    pool = mp.Pool(no_of_cpu_for_use)
    output= pool.starmap_async(integrate_cons, [(k,i,tm, p) for k, i in enumerate(index_array)]).get()
    
    #resizing tm array to remove points.
    tm=shorten_the_array(tm,step_size_for_storing);
    
    for l in output:
        #print(l[0]) 
        plot_dict[l[0]]["AR"]=l[1]
        plot_dict[l[0]]["zeb"]=l[2]
        plot_dict[l[0]]["snail"]=l[3]
        plot_dict[l[0]]["mir200"]=l[4]
        plot_dict[l[0]]["mir34"]=l[5]
        plot_dict[l[0]]["notch"]=l[6]
        plot_dict[l[0]]["delta"]=l[7]
        plot_dict[l[0]]["jagged"]=l[8]
        plot_dict[l[0]]["nicd"]=l[9]
        plot_dict[l[0]]["time"]=tm
        pool.close()

        ## Saving data:
    csv_filename = "lAtoZ=" + str(p["lAtoZ"]) + "_lZtoA=" + str(p["lZtoA"]) + "_Jt=" + str(p["Jt"]) +"_Dt=" + str(p["Dt"])
    np.save(folder+csv_filename,plot_dict)
        # Plotting in vaious subplots:
    npoints=len(tm)
    
    f, axs = plt.subplots(3,3,figsize=(15,7.5))        
    for key in plot_dict:
        axs[0,0].plot(tm[int(npoints/10):],plot_dict[key]["AR"][int(npoints/10):])
        axs[0,0].set_ylabel("AR values")
        axs[0,0].set_xlabel("time")

        axs[0,1].plot(tm[int(npoints/10):],plot_dict[key]["zeb"][int(npoints/10):])
        axs[0,1].set_ylabel("zeb values")
        axs[0,1].set_xlabel("time")

        axs[0,2].plot(tm[int(npoints/10):],plot_dict[key]["snail"][int(npoints/10):])
        axs[0,2].set_ylabel("snail values")
        axs[0,2].set_xlabel("time")
        
        axs[1,0].plot(tm[int(npoints/10):],plot_dict[key]["mir200"][int(npoints/10):])
        axs[1,0].set_ylabel("mir200 values")
        axs[1,0].set_xlabel("time")

        axs[1,1].plot(tm[int(npoints/10):],plot_dict[key]["mir34"][int(npoints/10):])
        axs[1,1].set_ylabel("mir34 values")
        axs[1,1].set_xlabel("time")
        
        axs[1,2].plot(tm[int(npoints/10):],plot_dict[key]["notch"][int(npoints/10):])
        axs[1,2].set_ylabel("notch values")
        axs[1,2].set_xlabel("time")

        axs[2,0].plot(tm[int(npoints/10):],plot_dict[key]["delta"][int(npoints/10):])
        axs[2,0].set_ylabel("delta values")
        axs[2,0].set_xlabel("time")

        axs[2,1].plot(tm[int(npoints/10):],plot_dict[key]["jagged"][int(npoints/10):])
        axs[2,1].set_ylabel("jagged values")
        axs[2,1].set_xlabel("time")

        axs[2,2].plot(tm[int(npoints/10):],plot_dict[key]["nicd"][int(npoints/10):])
        axs[2,2].set_ylabel("nicd values")
        axs[2,2].set_xlabel("time")

    f.suptitle(csv_filename)
    plt.subplots_adjust(top=0.9, bottom=0.2, left=0.10, right=0.95, hspace=0.25,wspace=0.5)
    plt.savefig(folder+csv_filename+".png",dpi=200)    
    plt.close()
    print("Simulation time(hrs):",(time.time()-t)/3600) 
