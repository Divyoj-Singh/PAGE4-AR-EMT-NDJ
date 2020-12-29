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

#%% ############ A to Z = 0.1 and Z to A = 0.9 #####################
folder="./dynamics/AtoZ=0.1_ZtoA=0.9/"
if not os.path.exists(folder):
    os.makedirs(folder)

Sic_array=[160000,200000,215000,240000,300000]


for Sic in Sic_array:
    
    p=aux.parameters(dt)
    p['lMtoA']=p['lAtoC']=0.1
    p['lAtoZ']=0.1
    p['lZtoA']=0.9
    p['Sic']=Sic
    aux.multiple_dynamic(tm,p,100,folder)

#%% ############ A to Z = 0.1 and Z to A = 0.1 #####################
folder="./dynamics/AtoZ=0.1_ZtoA=0.1/"
if not os.path.exists(folder):
    os.makedirs(folder)

Sic_array=[160000,185000,200000,215000,240000]
for Sic in Sic_array:
    
    p=aux.parameters(dt)
    p['lMtoA']=p['lAtoC']=0.1
    p['lAtoZ']=0.1
    p['lZtoA']=0.1
    p['Sic']=Sic
    aux.multiple_dynamic(tm,p,100,folder)
