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

#%% ############ A to Z = 0.9 and Z to A = 0.1 #####################
folder="./dynamics/AtoZ=0.9_ZtoA=0.1/"
if not os.path.exists(folder):
    os.makedirs(folder)

Sic_array=[160000,185000,195000,200000,215000,240000]

for Sic in Sic_array:
    
    p=aux.parameters(dt)
    p['lMtoA']=p['lAtoC']=0.1
    p['lAtoZ']=0.9
    p['lZtoA']=0.1
    p['Sic']=Sic
    aux.multiple_dynamic(tm,p,100,folder)