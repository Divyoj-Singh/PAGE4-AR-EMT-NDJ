# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 01:52:27 2021

@author: csb
"""

############################################
#%% importing packages:

import os
import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns
import time

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

def find_extremal(t,x):
    ''' function to find extremal points in oscillating trajectory'''   
    index = []
    tm = np.array([])
    rate = np.array([])
    n = t.size
    for i in np.arange(n/2, n-2, 1):   # 2
        i=int(i)   
        if ( x[i]>x[i+1] and x[i]>x[i-1] ):
            if ( x[i]>x[i+2] and x[i]>x[i-2] ):
                tm = np.append(tm, t[i])
                index.append(i)
                rate = np.append(rate, x[i])
    
        if ( x[i]<x[i+1] and x[i]<x[i-1] ):
            if ( x[i]<x[i+2] and x[i]<x[i-2] ):
                tm = np.append(tm, t[i])
                index.append(i)
                rate = np.append(rate, x[i])
    return tm, index, rate

#%% Euler Integration of ODES: 
def integrate_cons(i,time, p):
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
    return A,W

#%%
initial_t=time.time()

#%% Time scales:
T = 10*(24*7)
dt = 0.1
tm = np.arange(0,T+dt,dt)

#%% ############  #####################
# Sic_array=[160000,200000,215000,240000,300000]
folder="./Amplitude/AtoZ=0.1_ZtoA=0.1/Sic=160000/AR/"
if not os.path.exists(folder):
    os.makedirs(folder)
seed_value=1;

#%% Wild type case:
p=parameters(dt)
p['lMtoA']=p['lAtoC']=0.1
p['lAtoZ']=0.1
p['lZtoA']=0.1
p['Sic']=160000
# parameter_dictionary={}
# parameter_dictionary['Sic']=160000
# df=pd.read_csv(folder+'Amplitude_'+str(parameter_dictionary['Sic'])+'.csv')

A1,W1= integrate_cons(seed_value,tm, p)
tm1, index1, extr1 = find_extremal(tm, A1)
wt_amplitude=max(extr1)-min(extr1)


#%% Defining the dataframe to store percent changes:

df=pd.DataFrame(index=list(p.keys()),columns=['Parameter','+10%','-10%'])
df['Parameter']= df.index
df=df.drop(['Sic'],axis=0)

#%% Iterating over parameters:
for parameter_id in p:

    parameter_dictionary=p

    
    #%% Skip the micro-array interaction parameters:
    parameter=p[parameter_id]
    if type(parameter)== list:
        df=df.drop([parameter_id],axis=0)
        continue
    if parameter_id=='Sic':
        continue
    
    #%% plus 10%
    parameter_dictionary[parameter_id]=parameter+(parameter/10)

    A2,W2= integrate_cons(seed_value,tm, parameter_dictionary)
    tm2, index2, extr2 = find_extremal(tm, A2)
    plus_amplitude=max(extr2)-min(extr2)
    df.at[parameter_id,'+10%']=((plus_amplitude-wt_amplitude)/wt_amplitude)*100
    
    #%% minus 10%
    parameter=p[parameter_id]
    parameter_dictionary[parameter_id]=parameter-(parameter/10)

    A3,W3= integrate_cons(seed_value,tm, parameter_dictionary)
    tm3, index3, extr3 = find_extremal(tm, A3)
    minus_amplitude=max(extr3)-min(extr3)
    df.at[parameter_id,'-10%']=((minus_amplitude-wt_amplitude)/wt_amplitude)*100
    
    #%% plotting dynamics: 
    f, ax = plt.subplots(figsize=(3.6,2.7))
    sns.set_theme(style="whitegrid")        
    ax.plot(tm/(24*7),A1,'k-',label = 'wt')# , label = 'IC1'
    ax.plot(tm/(24*7),A2,'b-',label = '+10%') #, label = 'IC2'
    ax.plot(tm/(24*7),A3,'r-',label = '-10%') #, label = 'IC2'
    
    ax.tick_params(axis='y',labelsize=10,rotation=90)
    ax.tick_params(axis='x',labelsize=10)
    ax.set_ylabel("AR (Dimensionless)",fontsize=12)
    ax.set_xlabel("Time (in weeks)",fontsize=12)
    ax.set_title(str(parameter_id))       
    #plt.legend(loc = 'upper left', ncol=2)
    plt.subplots_adjust(top=0.80, bottom=0.20, left=0.20, right=0.95, hspace=0.25,wspace=0.5)
    plt.savefig(folder+str(parameter_id)+'.jpeg',dpi=200)
    plt.close()

        
df=df.sort_values(by=['+10%','-10%'],ascending=False)   
df.to_csv(folder+'Amplitude_'+str(parameter_dictionary['Sic'])+'.csv')    

#%% plotting Sensitivity plot:
sns.set_theme(style="white")

df.sort_values("+10%",ascending=False)
# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(3.5, 7))

# Plot
sns.set_color_codes("pastel")
ax=sns.barplot(x='+10%', y='Parameter', data=df, label="+10%", color="b")

ax=sns.barplot(x='-10%', y='Parameter', data=df, label="-10%", color="r")


ax.legend(ncol=2, loc="lower right", frameon=True)
sns.despine(left=True, bottom=True)

ax.tick_params(axis='y',labelsize=8)
ax.tick_params(axis='x',labelsize=8,rotation=45)

ax.set_ylabel("Parameters",fontsize=10)
ax.set_xlabel("% Change in Amplitude of AR",fontsize=10)
ax.set_title("Snail="+str(parameter_dictionary['Sic']))
plt.axvline(x=10,color='k', linestyle='--')
plt.axvline(x=-10,color='k', linestyle='--')      
ax.get_legend().remove()
plt.subplots_adjust(top=0.90, bottom=0.10, left=0.30, right=0.95, hspace=0.25,wspace=0.5)
plt.savefig(folder+'Amplitude_'+str(parameter_dictionary['Sic'])+'.jpeg',dpi=2000)

print("Simulation time(hrs):",(time.time()-initial_t)/3600)

