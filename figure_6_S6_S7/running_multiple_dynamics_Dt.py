# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 09:20:11 2020

@author: divyoj
"""
import auxiliary_functions as aux
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

#%% Time scales:
T = 10*(24*7)
dt = 0.001
tm = np.arange(0,T+dt,dt)

### Dt as the bifurcation paramter:
#%% ############ A to Z = 1 and Z to A = 1 #####################
folder="./dynamics/Dt/AtoZ=1_ZtoA=1/"
if not os.path.exists(folder):
    os.makedirs(folder)

Dt_array=[100,200,300,400,900]
for Dt in Dt_array:
    
    p=aux.parameters(dt)
    p['Nt']=1.0e+4;p['gD']=40;p['gJ']=15;
    p['lMtoA']=p['lAtoC']=0.1
    p['lZtoA']=1
    p['lAtoZ']=1
    p['Dt']=Dt
    aux.multiple_dynamic(tm,p,100,folder)
    

#%% ############ A to Z = 0.1 and Z to A = 0.9 #####################
folder="./dynamics/Dt/AtoZ=0.1_ZtoA=0.9/"
if not os.path.exists(folder):
    os.makedirs(folder)

Dt_array=[100,200,400,900]

for Dt in Dt_array:
    
    p=aux.parameters(dt)
    p['Nt']=1.0e+4;p['gD']=40;p['gJ']=15;
    p['lMtoA']=p['lAtoC']=0.1
    p['lAtoZ']=0.1
    p['lZtoA']=0.9
    p['Dt']=Dt
    aux.multiple_dynamic(tm,p,100,folder)

#%% ############ A to Z = 0.9 and Z to A = 0.1 #####################
folder="./dynamics/Dt/AtoZ=0.9_ZtoA=0.1/"
if not os.path.exists(folder):
    os.makedirs(folder)

Dt_array=[100,200,300,400,900]

for Dt in Dt_array:
    
    p=aux.parameters(dt)
    p['Nt']=1.0e+4;p['gD']=40;p['gJ']=15;
    p['lMtoA']=p['lAtoC']=0.1
    p['lAtoZ']=0.9
    p['lZtoA']=0.1
    p['Dt']=Dt
    aux.multiple_dynamic(tm,p,100,folder)

