# -*- coding: utf-8 -*-
"""
Created on Tue Sep  8 12:20:39 2020

@author: divyo
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns 

## indicating the folder:
input_folder='./dynamics/AtoZ=0.9_ZtoA=0.1/'
output_folder=input_folder+"plots/"
if not os.path.exists(output_folder+"AR/"):
    os.makedirs(output_folder+"AR/")
    
if not os.path.exists(output_folder+"mir200/"):
    os.makedirs(output_folder+"mir200/")
    
for filename in os.listdir(input_folder):
    if filename.endswith(".npy"):
        print(filename)
        plot_dict={}
        data=np.load(input_folder+filename,allow_pickle=True)
        index_array=data.item().keys()
        for index in index_array:
            plot_dict[index]={"AR":data.item().get(index)['AR'],"time":data.item().get(index)['time'],"mir200":data.item().get(index)['mir200']}
        del data;
        
        f, ax = plt.subplots(figsize=(3.6,2.7))        
        for key in plot_dict:
            tm=plot_dict[key]["time"]/(24*7)
            npoints=len(tm)
            ax.plot(tm,plot_dict[key]["AR"],linewidth=1)
        
        ax.tick_params(axis='y',labelsize=10,rotation=90)
        ax.tick_params(axis='x',labelsize=10)
        ax.set_ylim([0,300]) 
        ax.set_ylabel("AR (Dim.less)",fontsize=12)
        ax.set_xlabel("Time (in weeks)",fontsize=12)
            
               
       	#f.suptitle(filename[:-4])
       	plt.subplots_adjust(top=0.95, bottom=0.20, left=0.15, right=0.95, hspace=0.25,wspace=0.5)
       	plt.savefig(output_folder+"AR/AR_"+filename[:-4]+".png",dpi=2000)    
       	plt.close()
           
        f, ax = plt.subplots(figsize=(3.6,2.7))        
        for key in plot_dict:
            tm=plot_dict[key]["time"]/(24*7)
            npoints=len(tm)
            ax.plot(tm,(plot_dict[key]["mir200"]/1000),linewidth=1)
        
        ax.tick_params(axis='y',labelsize=10,rotation=90)
        ax.tick_params(axis='x',labelsize=10)
        
        ax.set_ylabel("mir200 (K molecules)",fontsize=12)
        ax.set_xlabel("Time (in weeks)",fontsize=12)
        ax.set_ylim([0,27])    
               
       	#f.suptitle(filename[:-4])
       	plt.subplots_adjust(top=0.95, bottom=0.20, left=0.15, right=0.95, hspace=0.25,wspace=0.5)
       	plt.savefig(output_folder+"mir200/mir200_"+filename[:-4]+".png",dpi=2000)    
       	plt.close()
   
           
