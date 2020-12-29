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

#%% ############ A to Z = 1 and Z to A = 1 ## for oscillations #####################
folder="./dynamics/AtoZ=1_ZtoA=1/"
if not os.path.exists(folder):
    os.makedirs(folder)

Sic_array=[200000]
for Sic in Sic_array:
    
    p=aux.parameters(dt)
    p['lMtoA']=p['lAtoC']=0.1 
    p['lZtoA']=1
    p['lAtoZ']=1
    p['Sic']=Sic
    aux.multiple_dynamic(tm,p,100,folder)

#%% ############ A to Z = 1 and Z to A = 1  # for monostable #####################
folder="./dynamics/AtoZ=1_ZtoA=1/"
if not os.path.exists(folder):
    os.makedirs(folder)

Sic_array=[200000]
for Sic in Sic_array:
    
    p=aux.parameters(dt)
    p['lMtoA']=p['lAtoC']=0.9 
    p['lZtoA']=1
    p['lAtoZ']=1
    p['Sic']=Sic
    aux.multiple_dynamic(tm,p,100,folder)

