# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 23:01:06 2021

@author: csb
"""


import auxiliary_functions as aux
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os

initial_time=time.time()

#%% Time scales:
T = 0.1*(24*7)
dt = 0.1
tm = np.arange(0,T+dt,dt)

#%% ############ Oscillations #####################

initial_time=time.time()
folder="./dynamics/"
if not os.path.exists(folder):
    os.makedirs(folder)


data = np.empty((0,6), int)
lAtoZ_array=np.around(np.linspace(0.0, 1.0, num=21, endpoint=True),2)
lZtoA_array=np.around(np.linspace(0.0, 1.0, num=21, endpoint=True),2)
nsim=10

for lAtoZ in lAtoZ_array:
    for lZtoA in lZtoA_array:
        
        p=aux.parameters(dt)
        p['lMtoA']=p['lAtoC']=0.1
    
        p['lAtoZ']=lAtoZ
        p['lZtoA']=lZtoA
        print(lAtoZ,lZtoA)
        p['Sic']=215000
        run_dictionary=aux.multiple_dynamic(tm,p,nsim,folder)
        
        for key in run_dictionary:
            n_point=len(run_dictionary[key]["mir200"])        
            mir200_level = np.mean(run_dictionary[key]["mir200"][int(n_point/2):])
            Ar_level =np.mean(run_dictionary[key]["AR"][int(n_point/2):])
            mir200_ampl = np.amax(run_dictionary[key]["mir200"][int(n_point/2):]) - np.amin(run_dictionary[key]["mir200"][int(n_point/2):])
            AR_ampl = np.amax(run_dictionary[key]["AR"][int(n_point/2):]) - np.amin(run_dictionary[key]["AR"][int(n_point/2):])
            arr=[lAtoZ,lZtoA,mir200_level,mir200_ampl,Ar_level,AR_ampl]
            data=np.vstack((data,arr))
            np.save("Snail="+str(p['Sic'])+".npy", data)
            # save to file (l1, l2, mir200_level, mir200_ampl, AR_level, AR_ampl)

np.save("Snail="+str(p['Sic'])+".npy", data)
# with this, you end up with an array with 6 columns and lAtoZ_array.size x lZtoA_array.size x nsim rows

print("Simulation time(hrs):",(time.time()-initial_time)/3600)          
