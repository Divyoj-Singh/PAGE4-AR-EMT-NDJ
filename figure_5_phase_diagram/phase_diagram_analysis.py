
# for data collection, I would just save the average and amplitude of miR200 and AR, as well as lambda values for each simulation
# fix snail
# fix a number of simulations for each parameetr combination (nsim=50 that you were already using could be a good number)

# try first with very few lambda values (say 3 or 5 at most) and very few trajectories (say nsim=2 or 3) to get a good time estimate and see how much parameter resolution we can afford.
# this will also help if there's any bug in the code!


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#e41a1c']#, '#e41a1c', '#dede00'

palette = sns.color_palette(["#f85033","#9056a3","#4c6c69","#2290cf","#6fa952","#ffff00","#ffa640"])
Sic="240000"
folder="./Sic="+Sic+"/"
filename="Snail="+Sic

#%%
data_array=np.load(folder+filename+".npy",allow_pickle=True)
lAtoZ_array=np.around(np.linspace(0.1, 1.0, num=21, endpoint=True),2)
lZtoA_array=np.around(np.linspace(0.1, 1.0, num=21, endpoint=True),2)
nsim=100

# 1) count fraction of E, E/M and M for each parameter combination
# for EMT, we are pretty confident that mir200=5000 and mir200=15000 are good boundaries to distinguish E, E/M and M

e_frac, em_frac, m_frac = np.zeros((lAtoZ_array.size, lZtoA_array.size)), np.zeros((lAtoZ_array.size, lZtoA_array.size)), np.zeros((lAtoZ_array.size, lZtoA_array.size))
e_ampl, em_ampl, m_ampl = np.zeros((lAtoZ_array.size, lZtoA_array.size)), np.zeros((lAtoZ_array.size, lZtoA_array.size)), np.zeros((lAtoZ_array.size, lZtoA_array.size))
mir200_avg = data_array[:,2] # data from big data file
mir200_ampl = data_array[:,3]
z = 0
for i in range(lAtoZ_array.size):
    for j in range(lZtoA_array.size):
        for n in range(nsim):
            if mir200_avg[z]<5000.:
                m_frac[i][j] = m_frac[i][j] + 1
                m_ampl[i][j] = m_ampl[i][j] + mir200_ampl[z]
            elif mir200_avg[z]<15000.:
                em_frac[i][j] = em_frac[i][j] + 1
                em_ampl[i][j] = em_ampl[i][j] + mir200_ampl[z]
            else:
                e_frac[i][j] = e_frac[i][j] + 1
                e_ampl[i][j] = e_ampl[i][j] + mir200_ampl[z]
            z = z + 1

        # let's normalize cell fractions and amplitude
        if m_frac[i][j]>0:
            m_ampl[i][j] = m_ampl[i][j]/m_frac[i][j]
            m_frac[i][j] = m_frac[i][j]/nsim
        if em_frac[i][j]>0:
            em_ampl[i][j] = em_ampl[i][j]/em_frac[i][j]
            em_frac[i][j] = em_frac[i][j]/nsim
        if e_frac[i][j]>0:
            e_ampl[i][j] = e_ampl[i][j]/e_frac[i][j]
            e_frac[i][j] = e_frac[i][j]/nsim
#%%            
# from here, we can construct a matrix with integer values to track multistability, i.e.
# only E: m = 0
# only E/M: m = 1
# only M: m = 2
# E + E/M: m = 3
# E + M: m = 4
# E/M + M: m = 5
# E + E/M + M: m = 6
# and plot is similar to figure 2b-c-d
state = np.zeros((lAtoZ_array.size, lZtoA_array.size))
for i in range(lAtoZ_array.size):
    for j in range(lZtoA_array.size):
        condition=np.array([e_frac[i][j],em_frac[i][j],m_frac[i][j]])>0.02 # threshold for fractions    
        if np.array_equal(condition, [True,False,False]):
            state[i][j]=0
        if np.array_equal(condition, [False,True,False]):
            state[i][j]=1
        if np.array_equal(condition, [False,False,True]):
            state[i][j]=2    
        if np.array_equal(condition, [True,True,False]):
            state[i][j]=3
        if np.array_equal(condition, [True,False,True]):
            state[i][j]=4
        if np.array_equal(condition, [False,True,True]):
            state[i][j]=5
        if np.array_equal(condition, [True,True,True]):
            state[i][j]=6
            
dataframe_2d=pd.DataFrame(data=state,index=lAtoZ_array,columns=lZtoA_array)
dataframe_2d=dataframe_2d[::-1] #reordering dataframe so that 0 is at left corner
f, ax = plt.subplots(figsize=(3.75,3.75))        
ax = sns.heatmap(dataframe_2d,cmap=palette,vmin=0,vmax=6,cbar=True)

## Fixing the xtick labels:
no_of_xticks=6;
ax.set_xticks(np.linspace(0,len(dataframe_2d.axes[1]),no_of_xticks)) ## removing any other points than the range:
new_xtick_labels=list(map(str,np.around(np.linspace(0.0,1.0,no_of_xticks),2))) 
## first made a linearly spaced array between min_N and max_N and then rounding it off
## then converting it to a string and then making a list of it.   
plt.xticks(np.linspace(0,len(dataframe_2d.axes[1]),no_of_xticks),new_xtick_labels)   

 ## Fixing the ytick labels:
no_of_yticks=6;
ax.set_yticks(np.linspace(0,len(dataframe_2d.axes[0]),no_of_yticks)) ## removing any other points than the range:
new_ytick_labels=list(map(str,np.around(np.linspace(1.0,0,no_of_yticks),2))) 
## first made a linearly spaced array between min_N and max_N and then rounding it off
## then converting it to a string and then making a list of it.   
plt.yticks(np.linspace(0,len(dataframe_2d.axes[0]),no_of_yticks),new_ytick_labels)

ax.tick_params(axis='y',labelsize=12)
ax.tick_params(axis='x',labelsize=12,rotation=0)
ax.set_ylabel("$\lambda_{AtoZ}$",fontsize=14)
ax.set_xlabel("$\lambda_{ZtoA}$",fontsize=14)
#ax.set_title(Sic+" state of mir200",fontsize=14)
       
plt.subplots_adjust(top=0.98, bottom=0.15, left=0.20, right=0.90, hspace=0.25,wspace=0.5)
plt.savefig(folder+filename+" State of mir200"+".jpeg",dpi=2000)    

#%% we can also construct three phase diagrams with the oscillation amplitude of E, E/M, and M for SI
## Epithelial
dataframe_2d=pd.DataFrame(data=e_ampl,index=lAtoZ_array,columns=lZtoA_array)
dataframe_2d=dataframe_2d[::-1] #reordering dataframe so that 0 is at left corner
f, ax = plt.subplots(figsize=(4,4))         
ax = sns.heatmap(dataframe_2d,cmap="RdYlBu",cbar=True)

## Fixing the xtick labels:
no_of_xticks=6;
ax.set_xticks(np.linspace(0,len(dataframe_2d.axes[1]),no_of_xticks)) ## removing any other points than the range:
new_xtick_labels=list(map(str,np.around(np.linspace(0.0,1.0,no_of_xticks),2))) 
## first made a linearly spaced array between min_N and max_N and then rounding it off
## then converting it to a string and then making a list of it.   
plt.xticks(np.linspace(0,len(dataframe_2d.axes[1]),no_of_xticks),new_xtick_labels)   

 ## Fixing the ytick labels:
no_of_yticks=6;
ax.set_yticks(np.linspace(0,len(dataframe_2d.axes[0]),no_of_yticks)) ## removing any other points than the range:
new_ytick_labels=list(map(str,np.around(np.linspace(1.0,0,no_of_yticks),2))) 
## first made a linearly spaced array between min_N and max_N and then rounding it off
## then converting it to a string and then making a list of it.   
plt.yticks(np.linspace(0,len(dataframe_2d.axes[0]),no_of_yticks),new_ytick_labels)

ax.tick_params(axis='y',labelsize=12)
ax.tick_params(axis='x',labelsize=12,rotation=0)
ax.set_ylabel("$\lambda_{AtoZ}$",fontsize=14)
ax.set_xlabel("$\lambda_{ZtoA}$",fontsize=14)
#ax.set_title(Sic+" Amplitude of Epithelial branch",fontsize=14)
       
plt.subplots_adjust(top=0.90, bottom=0.15, left=0.15, right=0.90, hspace=0.25,wspace=0.5)
plt.savefig(folder+filename+" Amplitude of Epithelial branch"+".jpeg",dpi=2000)

## Hybrid  ###########################################################################

dataframe_2d=pd.DataFrame(data=em_ampl,index=lAtoZ_array,columns=lZtoA_array)
dataframe_2d=dataframe_2d[::-1] #reordering dataframe so that 0 is at left corner
f, ax = plt.subplots(figsize=(4,4))        
ax = sns.heatmap(dataframe_2d,cmap="RdYlBu",cbar=True)

## Fixing the xtick labels:
no_of_xticks=6;
ax.set_xticks(np.linspace(0,len(dataframe_2d.axes[1]),no_of_xticks)) ## removing any other points than the range:
new_xtick_labels=list(map(str,np.around(np.linspace(0.0,1.0,no_of_xticks),2))) 
## first made a linearly spaced array between min_N and max_N and then rounding it off
## then converting it to a string and then making a list of it.   
plt.xticks(np.linspace(0,len(dataframe_2d.axes[1]),no_of_xticks),new_xtick_labels)   

 ## Fixing the ytick labels:
no_of_yticks=6;
ax.set_yticks(np.linspace(0,len(dataframe_2d.axes[0]),no_of_yticks)) ## removing any other points than the range:
new_ytick_labels=list(map(str,np.around(np.linspace(1.0,0,no_of_yticks),2))) 
## first made a linearly spaced array between min_N and max_N and then rounding it off
## then converting it to a string and then making a list of it.   
plt.yticks(np.linspace(0,len(dataframe_2d.axes[0]),no_of_yticks),new_ytick_labels)

ax.tick_params(axis='y',labelsize=12)
ax.tick_params(axis='x',labelsize=12,rotation=0)
ax.set_ylabel("$\lambda_{AtoZ}$",fontsize=14)
ax.set_xlabel("$\lambda_{ZtoA}$",fontsize=14)
#ax.set_title(Sic+" Amplitude of Hybrid branch",fontsize=14)
       
plt.subplots_adjust(top=0.90, bottom=0.15, left=0.15, right=0.95, hspace=0.25,wspace=0.5)
plt.savefig(folder+filename+" Amplitude of Hybrid branch"+".jpeg",dpi=2000)

## Mesenchymal  ###########################################################################

dataframe_2d=pd.DataFrame(data=m_ampl,index=lAtoZ_array,columns=lZtoA_array)
dataframe_2d=dataframe_2d[::-1] #reordering dataframe so that 0 is at left corner
f, ax = plt.subplots(figsize=(4,4))        
ax = sns.heatmap(dataframe_2d,cmap="RdYlBu",cbar=True)

## Fixing the xtick labels:
no_of_xticks=6;
ax.set_xticks(np.linspace(0,len(dataframe_2d.axes[1]),no_of_xticks)) ## removing any other points than the range:
new_xtick_labels=list(map(str,np.around(np.linspace(0.0,1.0,no_of_xticks),2))) 
## first made a linearly spaced array between min_N and max_N and then rounding it off
## then converting it to a string and then making a list of it.   
#plt.xticks(np.linspace(0,len(dataframe_2d.axes[1]),no_of_xticks),new_xtick_labels)   

 ## Fixing the ytick labels:
no_of_yticks=6;
ax.set_yticks(np.linspace(0,len(dataframe_2d.axes[0]),no_of_yticks)) ## removing any other points than the range:
new_ytick_labels=list(map(str,np.around(np.linspace(1.0,0,no_of_yticks),2))) 
## first made a linearly spaced array between min_N and max_N and then rounding it off
## then converting it to a string and then making a list of it.   
plt.yticks(np.linspace(0,len(dataframe_2d.axes[0]),no_of_yticks),new_ytick_labels)

ax.tick_params(axis='y',labelsize=12)
ax.tick_params(axis='x',labelsize=12,rotation=0)
ax.set_ylabel("$\lambda_{AtoZ}$",fontsize=14)
ax.set_xlabel("$\lambda_{ZtoA}$",fontsize=14)
ax.set_title(Sic+" Amplitude of Mesenchymal branch",fontsize=14)
       
plt.subplots_adjust(top=0.90, bottom=0.15, left=0.15, right=0.95, hspace=0.25,wspace=0.5)
plt.savefig(folder+filename+" Amplitude of Mesenchymal branch"+".jpeg",dpi=2000)    

#%%
# 2) region of oscillation of AR
# for AR, since we really don't know the values for high AR or low AR, we could first just try to identify regions of oscillations and regions without oscillations
osc_frac = np.zeros((lAtoZ_array.size, lZtoA_array.size)) # fraction of trajectories that oscillate
osc_ampl = np.zeros((lAtoZ_array.size, lZtoA_array.size)) # oscillation amplitude (for the fraction of trajectories that oscillate)
AR_ampl = data_array[:,5]
AR_avg = data_array[:,5]
z = 0
epsilon=0.1 ## looking at the trajectories, this looks reasonable
for i in range(lAtoZ_array.size):
    for j in range(lZtoA_array.size):
        for n in range(nsim):
            if (AR_ampl[z]/AR_avg[z])>epsilon:
                osc_frac[i][j] = osc_frac[i][j] + 1
                osc_ampl[i][j] = osc_ampl[i][j] + AR_ampl[z]
            z = z + 1
        if osc_frac[i][j]>0:
            osc_ampl[i][j] = osc_ampl[i][j]/osc_frac[i][j]

osc_frac = osc_frac/nsim

#%% now osc_frac is the probability to oscillate as a function of lAtoZ_array, lZtoA_array, and we can plot it as a phase diagram.


dataframe_2d=pd.DataFrame(data=osc_frac,index=lAtoZ_array,columns=lZtoA_array)
dataframe_2d=dataframe_2d[::-1] #reordering dataframe so that 0 is at left corner
f, ax = plt.subplots(figsize=(4,4))        
ax = sns.heatmap(dataframe_2d,cmap="RdYlBu",cbar=True)

## Fixing the xtick labels:
no_of_xticks=6;
ax.set_xticks(np.linspace(0,len(dataframe_2d.axes[1]),no_of_xticks)) ## removing any other points than the range:
new_xtick_labels=list(map(str,np.around(np.linspace(0.0,1.0,no_of_xticks),2))) 
## first made a linearly spaced array between min_N and max_N and then rounding it off
## then converting it to a string and then making a list of it.   
plt.xticks(np.linspace(0,len(dataframe_2d.axes[1]),no_of_xticks),new_xtick_labels)   

 ## Fixing the ytick labels:
no_of_yticks=6;
ax.set_yticks(np.linspace(0,len(dataframe_2d.axes[0]),no_of_yticks)) ## removing any other points than the range:
new_ytick_labels=list(map(str,np.around(np.linspace(1.0,0,no_of_yticks),2))) 
## first made a linearly spaced array between min_N and max_N and then rounding it off
## then converting it to a string and then making a list of it.   
plt.yticks(np.linspace(0,len(dataframe_2d.axes[0]),no_of_yticks),new_ytick_labels)

ax.tick_params(axis='y',labelsize=12)
ax.tick_params(axis='x',labelsize=12,rotation=0)
ax.set_ylabel("$\lambda_{AtoZ}$",fontsize=14)
ax.set_xlabel("$\lambda_{ZtoA}$",fontsize=14)
#ax.set_title(Sic+" Fraction of cases where AR Oscillates",fontsize=14)
       
plt.subplots_adjust(top=0.90, bottom=0.15, left=0.15, right=0.95, hspace=0.25,wspace=0.5)
plt.savefig(folder+filename+" Fraction of cases where AR Oscillates"+".jpeg",dpi=2000)

#%% You can also make an additional phase plot with the oscillation amplitude

dataframe_2d=pd.DataFrame(data=osc_ampl,index=lAtoZ_array,columns=lZtoA_array)
dataframe_2d=dataframe_2d[::-1] #reordering dataframe so that 0 is at left corner
f, ax = plt.subplots(figsize=(3.75,3.75))        
ax = sns.heatmap(dataframe_2d,cmap="RdYlBu",cbar=True)

## Fixing the xtick labels:
no_of_xticks=6;
ax.set_xticks(np.linspace(0,len(dataframe_2d.axes[1]),no_of_xticks)) ## removing any other points than the range:
new_xtick_labels=list(map(str,np.around(np.linspace(0.0,1.0,no_of_xticks),2))) 
## first made a linearly spaced array between min_N and max_N and then rounding it off
## then converting it to a string and then making a list of it.   
plt.xticks(np.linspace(0,len(dataframe_2d.axes[1]),no_of_xticks),new_xtick_labels)   

 ## Fixing the ytick labels:
no_of_yticks=6;
ax.set_yticks(np.linspace(0,len(dataframe_2d.axes[0]),no_of_yticks)) ## removing any other points than the range:
new_ytick_labels=list(map(str,np.around(np.linspace(1.0,0,no_of_yticks),2))) 
## first made a linearly spaced array between min_N and max_N and then rounding it off
## then converting it to a string and then making a list of it.   
plt.yticks(np.linspace(0,len(dataframe_2d.axes[0]),no_of_yticks),new_ytick_labels)

ax.tick_params(axis='y',labelsize=12)
ax.tick_params(axis='x',labelsize=12,rotation=0)
ax.set_ylabel("$\lambda_{AtoZ}$",fontsize=14)
ax.set_xlabel("$\lambda_{ZtoA}$",fontsize=14)
#ax.set_title(Sic+" Amplitude of AR oscillation",fontsize=14)
       
plt.subplots_adjust(top=0.90, bottom=0.15, left=0.20, right=0.95, hspace=0.25,wspace=0.5)
plt.savefig(folder+filename+" Amplitude of AR oscillation"+".jpeg",dpi=2000)    