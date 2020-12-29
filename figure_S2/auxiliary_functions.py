# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 01:32:23 2020

@author: Divyoj Singh
"""
#  Auxilary functions.
# This file has the necessary functions to draw bifurcation diagram for EMT-AR-Page-4 network. Its in Python3.
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
import time
#%% Control parametes:
# Specify the number of cpu-cores to be used: 
no_of_cpu_for_use=40
step_size_for_storing=100; ## in dynamic_behaviour
# specify the number of initial conditions to start with, for finding the steady states
no_of_initial_conditions_for_bifurcation=100
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
           
            ## micro-array interaction:
           ,'u0200' : 1.0e+4
           ,'gu':[0.0e+0, 1*5.0e-3, 2*5.0e-2, 3*5.0e-1, 4*5.0e-1, 5*5.0e-1, 6*5.0e-1]
           ,'gm':[0.0e+0, 4.0e-2, 2.0e-1, 1.0e+0, 1.0e+0, 1.0e+0, 1.0e+0]
           ,'l':[1.0e+0, 6.0e-1, 3.0e-1, 1.0e-1, 5.0e-2, 5.0e-2, 5.0e-2]
           ,'km' : 5.0e-1,'kP' : 1.0e+2


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
            ## bifurcation parameter
            ,'Sic':200000 # Snail value
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
    # function to intergrate the differential equations over time and return the array of levels at each time point.

    dt = time[1]-time[0]
    npoints = time.size - 1 #int(T/dt)

    # define vectors to store results
    PU = np.empty(npoints+1);PM = np.empty(npoints+1);
    PH = np.empty(npoints+1);C = np.empty(npoints+1);A = np.empty(npoints+1);
    Z = np.empty(npoints+1);W = np.empty(npoints+1);S = np.empty(npoints+1);

    # setting the seed value for random number generation:
    random.seed(i)
    # setting the initial conditions:
    PM[0]=0.;PH[0]=0.;C[0]=0.;PU[0]=0.;
    W[0]=random.randint(0,100000);Z[0]=random.randint(0,5000000);S[0]=p['Sic']; 
    A[0]=random.randint(0,300);
    
    # integrating over time:
    for i in range(1,npoints+1):
        PU[i] = PU[i-1] + dt*( p['gP'] - p['H']*PU[i-1]/(PU[i-1]+p['a']) - p['gammaPU']*PU[i-1] ) # WT-page4
        PM[i] = PM[i-1] + dt*( p['H']*PU[i-1]/(PU[i-1]+p['a']) - p['gH']*C[i-1]*PM[i-1]/(PM[i-1]+p['b']) - p['gammaPM']*PM[i-1] )#HIPK1-PAGE4
        PH[i] = PH[i-1] + dt*( p['gH']*C[i-1]*PM[i-1]/(PM[i-1]+p['b']) - p['gammaPH']*PH[i-1] )#CLK2-PAGE4 (PH)
        C[i] = C[i-1] + dt*( p['gC']*HS(A[int(i-1-p['tau_C']*step(i,p['tau_C']))],p['A0'],p['nC'],p['lAtoC'])-p['gammaC']*C[i-1])# CLK2
 
        A[i] = A[i-1] + dt*( p['gA']*HS(Z[i-1],p['Z0A'],p['nZtoA'],p['lZtoA'])*HS(PM[int(i-1-p['tau_A']*step(i,p['tau_A']))],p['P0_A'],p['nA'],p['lMtoA']) - p['gammaA']*A[i-1] ) # Androgen
        Z[i] = Z[i-1] + dt*( p['kP']*p['gZ']*HS(A[i-1],p['A0Z'],p['nAtoZ'],p['lAtoZ'])*HS(Z[i-1],p['Z0Z'],p['nZZ'],p['lZZ'])*HS(S[i-1],p['S0Z'],p['nSZ'],p['lSZ'])*Pl(W[i-1],p['km'],6,p['u0200'],p['l'],p['gm']) - p['gammaZ']*Z[i-1] ) #Zeb
        W[i] = W[i-1] + dt*(p['gu200']*HS(Z[i-1],p['Z0u200'],p['nZu200'],p['lZu200'])*HS(S[i-1],p['S0u200'],p['nSu200'],p['lSu200']) - p['gZ']*HS(Z[i-1],p['Z0Z'],p['nZZ'],p['lZZ'])*HS(S[i-1],p['S0Z'],p['nSZ'],p['lSZ'])*Py(W[i-1],p['km'],6,p['u0200'],p['gu'],p['gm']) - p['gammau200']*W[i-1]) ## mir-200 
        S[i] =S[i-1]

    A=shorten_the_array(A,step_size_for_storing);
    W=shorten_the_array(W,step_size_for_storing);
                            
    return k,A,W

def steady_state(k,i,time, p):
    # function to intergrate the differential equations over time and return just the steady states:

    dt = time[1]-time[0]  #0.1
    npoints = time.size - 1 #int(T/dt)

    # define vectors to store results
    PU = np.empty(npoints+1);PM = np.empty(npoints+1);
    PH = np.empty(npoints+1);C = np.empty(npoints+1);A = np.empty(npoints+1);
    Z = np.empty(npoints+1);W = np.empty(npoints+1);S = np.empty(npoints+1);

    # setting the seed value for random number generation:
    random.seed(i)
    # setting the initial conditions:
    PM[0]=0.;PH[0]=0.;C[0]=0.;PU[0]=0.;
    W[0]=random.randint(0,100000);Z[0]=random.randint(0,5000000);S[0]=p['Sic']; 
    A[0]=random.randint(0,300);   
    # integrating over time:
    for i in range(1,npoints+1):
        PU[i] = PU[i-1] + dt*( p['gP'] - p['H']*PU[i-1]/(PU[i-1]+p['a']) - p['gammaPU']*PU[i-1] ) # WT-page4
        PM[i] = PM[i-1] + dt*( p['H']*PU[i-1]/(PU[i-1]+p['a']) - p['gH']*C[i-1]*PM[i-1]/(PM[i-1]+p['b']) - p['gammaPM']*PM[i-1] )#HIPK1-PAGE4
        PH[i] = PH[i-1] + dt*( p['gH']*C[i-1]*PM[i-1]/(PM[i-1]+p['b']) - p['gammaPH']*PH[i-1] )#CLK2-PAGE4 (PH)
        C[i] = C[i-1] + dt*( p['gC']*HS(A[int(i-1-p['tau_C']*step(i,p['tau_C']))],p['A0'],p['nC'],p['lAtoC'])-p['gammaC']*C[i-1])# CLK2
 
        A[i] = A[i-1] + dt*( p['gA']*HS(Z[i-1],p['Z0A'],p['nZtoA'],p['lZtoA'])*HS(PM[int(i-1-p['tau_A']*step(i,p['tau_A']))],p['P0_A'],p['nA'],p['lMtoA']) - p['gammaA']*A[i-1] ) # Androgen
        Z[i] = Z[i-1] + dt*( p['kP']*p['gZ']*HS(A[i-1],p['A0Z'],p['nAtoZ'],p['lAtoZ'])*HS(Z[i-1],p['Z0Z'],p['nZZ'],p['lZZ'])*HS(S[i-1],p['S0Z'],p['nSZ'],p['lSZ'])*Pl(W[i-1],p['km'],6,p['u0200'],p['l'],p['gm']) - p['gammaZ']*Z[i-1] ) #Zeb
        W[i] = W[i-1] + dt*(p['gu200']*HS(Z[i-1],p['Z0u200'],p['nZu200'],p['lZu200'])*HS(S[i-1],p['S0u200'],p['nSu200'],p['lSu200']) - p['gZ']*HS(Z[i-1],p['Z0Z'],p['nZZ'],p['lZZ'])*HS(S[i-1],p['S0Z'],p['nSZ'],p['lSZ'])*Py(W[i-1],p['km'],6,p['u0200'],p['gu'],p['gm']) - p['gammau200']*W[i-1]) ## mir-200 
        S[i] = S[i-1]
       
    return k,A[-1],W[-1],Z[-1]

def different_states(time,p):
    AR=[];mir200=[];Zeb=[];
    # index array contains the seed value for choosing different initial conditions:
    index_array = np.arange(1,no_of_initial_conditions_for_bifurcation,1)
    temp_AR=[];temp_mir200=[];temp_Zeb=[];
    # starting the pool of CPUs
    pool = mp.Pool(no_of_cpu_for_use)
    # parallelising function "steady state" over the index array and geeting the steady states back in output:
    output= pool.starmap_async(steady_state, [(k,i,time, p) for k, i in enumerate(index_array)]).get()
    
    for l in output:

        temp_AR.append(l[1])
        temp_mir200.append(l[2])
        temp_Zeb.append(l[3])
        pool.close()
    # rounding off the steady states values to reasonable orders: 
    temp_AR=np.around(temp_AR,0);AR=list(set(temp_AR))
    temp_mir200=np.around(temp_mir200,-1);mir200=list(set(temp_mir200))
    temp_Zeb=np.around(temp_Zeb,-1);Zeb=list(set(temp_Zeb))
    
    return AR,mir200,Zeb

def numerical_bifurcation(bifur_para,bifur_para_array,tm,p):
    t=time.time()
    bifur_dict={}
    for i in bifur_para_array:
        p[bifur_para]=i
        bifur_dict[i]={"Zeb":[],"AR":[],"mir200":[]}
        bifur_dict[i]["AR"],bifur_dict[i]["mir200"],bifur_dict[i]["Zeb"]=  different_states(tm, p)

    ## Saving data:
    csv_filename="lAtoC="+str(p["lAtoC"])+"_lMtoA="+str(p["lMtoA"])+ "_lAtoZ="+str(p["lAtoZ"])+"_lZtoA="+str(p["lZtoA"])+"_" + bifur_para + "=" + str(bifur_para_array[0]) + "_" + str(bifur_para_array[-1])
    np.save(csv_filename,bifur_dict)
    dff=pd.DataFrame.from_dict(bifur_dict,orient='index')
    dff.to_csv(csv_filename + ".csv")        
    ### plotting:    
    f, axs = plt.subplots(1,3,figsize=(15,2.5)) 
    for key in bifur_dict:
        sns.scatterplot(x=key,y=bifur_dict[key]["AR"],ax=axs[0])
        axs[0].set_ylabel("AR values")
        axs[0].set_xlabel("Snail Values")

        sns.scatterplot(x=key,y=bifur_dict[key]["mir200"],ax=axs[1])
        axs[1].set_ylabel("mir200 values")
        axs[1].set_xlabel("Snail Values")

        sns.scatterplot(x=key,y=bifur_dict[key]["Zeb"],ax=axs[2])
        axs[2].set_ylabel("Zeb values")
        axs[2].set_xlabel("Snail Values")
        
        
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.5)       
    f.suptitle(csv_filename)
    plt.savefig(csv_filename+".png",dpi=1000)
    plt.show()
    print("Simulation time(hrs):",(time.time()-t)/3600)
    
def multiple_dynamic(tm,p,n,folder):
    t=time.time()
    plot_dict={}
    index_array=np.arange(0,n,1)	
    for i in index_array:
        plot_dict[i]={"AR":[],"mir200":[],"time":[]}
    pool = mp.Pool(no_of_cpu_for_use)
    output= pool.starmap_async(integrate_cons, [(k,i,tm, p) for k, i in enumerate(index_array)]).get()
    
    #resizing tm array to remove points.
    tm=shorten_the_array(tm,step_size_for_storing);
    
    for l in output:
        plot_dict[l[0]]["AR"]=l[1]
        plot_dict[l[0]]["mir200"]=l[2]
        plot_dict[l[0]]["time"]=tm
        pool.close()

    ## Saving data:
    csv_filename="dynamics"+"_lAtoC="+str(p["lAtoC"])+"_lMtoA="+str(p["lMtoA"])+ "_lAtoZ="+str(p["lAtoZ"])+"_lZtoA="+str(p["lZtoA"])+"_Sic=" + str(p['Sic'])
    np.save(folder+csv_filename,plot_dict)    
    n_point=len(tm)
    #[int(n_point/32):]         
    # Plotting in vaious subplots:
    f, axs = plt.subplots(1,2,figsize=(15,5))        
    for key in plot_dict:
        axs[0].plot(tm,plot_dict[key]["AR"])
        axs[0].set_ylabel("AR")
        axs[0].set_xlabel("time (hours)")
        

        axs[1].plot(tm,(plot_dict[key]["mir200"]/1000))
        axs[1].set_ylabel("mir200 (in 10^3)")
        axs[1].set_xlabel("time (hours)")
    
    axs[0].tick_params(axis='y',labelsize=10,rotation=90)
    axs[1].tick_params(axis='y',labelsize=10,rotation=90)
    
    axs[0].set_ylim([0,300])
    axs[1].set_ylim([0,27])    
    
    #f.suptitle(csv_filename)
    
    plt.subplots_adjust(top=0.9, bottom=0.2, left=0.1, right=0.9, hspace=0.1,wspace=0.5)
    plt.savefig(folder+csv_filename+".png",dpi=1000)
    #plt.show()
    plt.close()
    print("Simulation time(hrs):",(time.time()-t)/3600)
