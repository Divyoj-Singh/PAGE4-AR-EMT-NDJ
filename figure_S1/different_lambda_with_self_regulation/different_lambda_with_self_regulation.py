# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 2020

@author: divyoj
"""
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import seaborn as sns
import pandas as pd 
import os
#%%
no_of_cpu_for_use=51
cores=mp.cpu_count()
print("total no of  cpu cores:",cores)
print("running on",no_of_cpu_for_use)

## making folder for data and graphs
if not os.path.exists("./data"):
    os.makedirs("./data")
if not os.path.exists("./plots"):
    os.makedirs("./plots")

#%%
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

#%%
# define shifted Hill function
def HS(x,x0,n,l):
    return (1+l*((x/x0)**n))/(1+((x/x0)**n))

# define step function for delay term
def step(x,tau):
    return 1 * (x > tau)

def integrate_cons(time, p, Aic, Xic):
    '''  function to integrate the page4+DNFL system'''
    dt = time[1]-time[0]  #0.1
    npoints = time.size - 1 #int(T/dt)
    # define vectors to store results
    PU = np.empty(npoints+1)
    PM = np.empty(npoints+1) # HIPK1-PAGE4 half-life 150 h
    PH = np.empty(npoints+1) # CLK2-PAGE4 10x less stable than
    C = np.empty(npoints+1)
    A = np.empty(npoints+1)
    X = np.empty(npoints+1)    
    PM[0]=0.; PH[0]=0.; C[0]=0.;
    A[0]=Aic; PU[0]=0.; X[0]=Xic
    for i in range(1,npoints+1):
        PU[i] = PU[i-1] + dt*( p['gP'] - p['H']*PU[i-1]/(PU[i-1]+p['a']) - p['kPU']*PU[i-1] )
        PM[i] = PM[i-1] + dt*( p['H']*PU[i-1]/(PU[i-1]+p['a']) - p['gH']*C[i-1]*PM[i-1]/(PM[i-1]+p['b']) - p['kPM']*PM[i-1] )
        PH[i] = PH[i-1] + dt*( p['gH']*C[i-1]*PM[i-1]/(PM[i-1]+p['b']) - p['kPH']*PH[i-1] )
        A[i] = A[i-1] + dt*( p['gA']*HS(X[i-1],p['X0'],p['n'],p['lXtoA'])*HS(PM[int(i-1-p['tau_A']*step(i,p['tau_A']))],p['P0_A'],p['nA'],p['lMtoA']) - p['kA']*A[i-1] )
        C[i] = C[i-1] + dt*( p['gC']*HS(A[int(i-1-p['tau_C']*step(i,p['tau_C']))],p['A0'],p['nC'],p['lAtoC']) - p['kC']*C[i-1] )
        X[i] = X[i-1] + dt*( p['gX']*HS(A[i-1],p['A0'],p['n'],p['lAtoX'])*HS(X[i-1],p['X0'],p['nX'],p['lXtoX']) - p['kX']*X[i-1] )
    return PM,PH,C,PU,A,X


    
def finding_state(k,lambda2,time, p):
    p['lXtoA'] = lambda2
    # consider two trajectories staring from opposite iniital conditions to check bistability
    PM1,PH1,C1,PU1,A1,X1 = integrate_cons(time, p, 300, 0)
    tm1, index1, extr1 = find_extremal(time, A1)
    PM2,PH2,C2,PU2,A2,X2 = integrate_cons(time, p, 0, 300)
    
    # if there are more than 2 extremal points -> possibility of oscillations
    if extr1.size > 2:
        amplitude1 = abs( extr1[-1]-extr1[-2] )/( extr1[-1]+extr1[-2] )
        # if oscillation is really a thing -> oscillations
        if amplitude1 > 0.1:
            amplitude = 6
            # else, if A1 and A2 differ -> bistability
        elif abs( A1[-1]-A2[-1] )/( A1[-1]+A2[-1] )>0.1:
            amplitude = 2
            # otherwise -> monostable
        else:
            amplitude= 1
        # less or equal than 2 extremal points -> not oscillating
    else:
        # if A1 and A2 differ -> bistable, otherwise monostable
        if abs( A1[-1]-A2[-1] )/( A1[-1]+A2[-1] )>0.1:
            amplitude = 2
        else:
            amplitude = 1
    #print('amplitude lMtoA AtoX',amplitude, p['lMtoA'],p['AtoX'])        
    return k,amplitude

#%% Specifying the time domain
T = 10*(24*7) ## simulating for 10 weeks:
dt = 0.001
tm = np.arange(0,T+dt,dt)

# compared to the original model, there is an additional variable (AR)
# to explicitly couple the double negative loop with the Page4 circuit


## model's parameters
# some parameters are slightly modified to include corrections for the additional interactions
p = {}
gamma_H = np.log(2)/15 # this is the degradation rate of H by which other paramters were rescaled. 
p['H']=50*gamma_H  # original H=10 because there was no loss due to chemical reaction

## Production rates:
p['gA'] = 100.0*gamma_H ## 4.62
p['gP'] = 50.0*gamma_H ## 2.31
p['gC']=60*gamma_H ## 2.77
# in addition: the conversion of HIPK1-P4 to CLK2-P4 originally had a rate ( gH*C*Pm/(Pm + B) in eqs. 7-8 of methods)
# after the rescaling, this parameter is equal to 1 and therefore no longer present (due to the choice of C0 in the SI )
# so, please introduce a new parameter that multiplies this term in the equations of PM and PH. Since this is a rate, it must be multiplied by gamma_H (so, it will just be equal to gamma_H)
# basically, every term in the page4 equations is rescaled by gamma_H, so you should get the same exact dynamics but on a new timescale 1/gamma_H
p['gH']=1*gamma_H ## 0.04

## Degradation rates: ## rescaled with np.log(2)/15  
p['kC']=0.5*gamma_H  ## 0.02
p['kPM']= 0.1*gamma_H # np.log(2)/150. #   0.004
p['kPH']= 1.0*gamma_H # np.log(2)/15. 0.04  
p['kPU'] = 0.4*gamma_H # np.log(2)/60., # 0.016
p['kA'] = 0.5*gamma_H # 0.020

## Threshold comstants:
p['a']=5
p['b']=20
# Delay terms:
# the other thing to change is the delay term dic[tau]=1.5/(2dt)
# here, tau is not defined as an actual time but rather as the number of timesteps. The value of tau in terms of time units is 1.5/2 = 0.75
# the real value of tau (in hours) is therefore: dic['tau'] = 0.75/gamma_H  - that's around 16 hours
# after that, you can divide tau by the timestep dt that you are using in the simulation (which has dimensions of time), and have again a tau that is defined as a number of integration steps 
# dic['tau'] = dic['tau']/dt (rememebr that you will need to change tau if you change the timestep dt!)

p['tau_A'] = 1.5/(2*dt*gamma_H)
p['tau_C'] = 1.5/(2*dt*gamma_H)

## Other paramters for interactions:
p['nA'] = 4
p['P0_A']=20
p['A0'] = 65
p['nC']=4
p['P0']=20

### lambdas of hill functions coupling AR with the Page4 circuit
p['lMtoA'] = 0.1 # H-P4 to AR
p['lAtoC'] = 0.1 # AR to CLK2
    
# parameters for additional node (x) for duble negative feedback
p['gX'] = 50*gamma_H  #2.31
p['kX'] = 1*gamma_H #0.04
p['n'] = 4
p['lAtoX'] = 0.9  # AR to x
p['lXtoA'] = 0.9 # x to AR
p['X0'] = 25.
#### self activation:
p['nX']=4
p['lXtoX']=1

#%% #############################################
lambda_array=[0.1,0.5,0.9]
for n in lambda_array:
    for m in lambda_array:
        p['lAtoC']=m
        p['lMtoA']=n
        filename="self_regul"+str(p['lXtoX'])+"AtoC="+ str(p['lAtoC']) + "_CtoA="+ str(p['lMtoA'])
        v1 = np.arange(0.0,2.01,0.04)
        v1=np.around(v1,2)
        v2 = np.arange(0.0,2.01,0.04)
        v2=np.around(v2,2)
        ampl = np.zeros((v1.size, v2.size))

        for i in range( v1.size ):
            print(i)
            p['lAtoX'] = v1[i]
            pool = mp.Pool(no_of_cpu_for_use)
            output= pool.starmap_async(finding_state, [(k, j, tm, p) for k, j in enumerate(v2)]).get()
            for l in output:
                ampl[i][l[0]]=l[1] 
                pool.close()
                
        phase_plot_data_frame=pd.DataFrame(data=ampl,columns=v2,index=v1)
        phase_plot_data_frame=phase_plot_data_frame[::-1]
        #%% Save data and plot:
        # save the data to a csv file:
        phase_plot_data_frame.to_csv("./data/"+filename+".csv")## wrting it
    
        phase_plot_data_frame.to_csv("./data/"+filename+".csv")
        ax = sns.heatmap(phase_plot_data_frame,cmap="RdYlBu_r",vmin=1, vmax=6)
        ax.set_xlabel("lamda XtoA")
        ax.set_ylabel("lambda AtoZ")
        ax.set_title("6-Oscillations,2-Bistable,1-Monostable"+'\n'+filename)
        plt.savefig('./plots/'+filename+'.png')
        plt.close()
