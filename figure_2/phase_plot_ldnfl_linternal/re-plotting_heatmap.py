# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 19:23:17 2020

@author: divyo
"""
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import seaborn as sns
import pandas as pd 
import os
CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']
   
folder='./'
for filename in os.listdir(folder+"data/"):
    print(filename)
    dataframe_2d = pd.read_csv(folder+"data/"+filename,engine='python',index_col=0)
    dataframe_2d=dataframe_2d.replace(2,3.5) # replacing 2 with 3.5 ((6+1)/2) so that its at the center of the cmap
    f, ax = plt.subplots(figsize=(5,5))        
    ax = sns.heatmap(dataframe_2d,center=3.5,cmap=CB_color_cycle,vmin=1, vmax=6,cbar=False)
        
    
    
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
    ax.set_ylabel("$\lambda_{PAGE4}$",fontsize=14)
    ax.set_xlabel("$\lambda_{DNFL}$",fontsize=14)
           
    plt.subplots_adjust(top=0.95, bottom=0.15, left=0.15, right=0.95, hspace=0.25,wspace=0.5)
    plt.savefig(filename[:-4]+".jpeg",dpi=2000)    
    #plt.close()
