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
input_folder='./dynamics/Jt/AtoZ=0.9_ZtoA=0.9/'
output_folder='./dynamics/'
for filename in os.listdir(input_folder):
    if filename.endswith(".npy"):
        print(filename)
        plot_dict={}
        data=np.load(input_folder+filename,allow_pickle=True)
        index_array=data.item().keys()
        for index in index_array:
            plot_dict[index]={"AR":data.item().get(index)['AR'],"time":data.item().get(index)['time'],"zeb":data.item().get(index)['zeb'],"snail":data.item().get(index)['snail'],"mir200":data.item().get(index)['mir200'],"mir34":data.item().get(index)['mir34'],"notch":data.item().get(index)['notch'],"jagged":data.item().get(index)['jagged'],"delta":data.item().get(index)['delta'],"nicd":data.item().get(index)['nicd']}
        del data;
        
        f, axs = plt.subplots(3,3,figsize=(15,7.5))        
        for key in plot_dict:
            tm=plot_dict[key]["time"]
            npoints=len(tm)
            axs[0,0].plot(tm[int(npoints/8):],plot_dict[key]["AR"][int(npoints/8):])
            axs[0,0].set_ylabel("AR values")
            axs[0,0].set_xlabel("time")
            
            axs[0,1].plot(tm[int(npoints/8):],plot_dict[key]["zeb"][int(npoints/8):])
            axs[0,1].set_ylabel("zeb values")
            axs[0,1].set_xlabel("time")
               
            axs[0,2].plot(tm[int(npoints/8):],plot_dict[key]["snail"][int(npoints/8):])
            axs[0,2].set_ylabel("snail values")
            axs[0,2].set_xlabel("time")
           
            axs[1,0].plot(tm[int(npoints/8):],plot_dict[key]["mir200"][int(npoints/8):])
            axs[1,0].set_ylabel("mir200 values")
            axs[1,0].set_xlabel("time")
   
            axs[1,1].plot(tm[int(npoints/8):],plot_dict[key]["mir34"][int(npoints/8):])
            axs[1,1].set_ylabel("mir34 values")
            axs[1,1].set_xlabel("time")
           
            axs[1,2].plot(tm[int(npoints/8):],plot_dict[key]["notch"][int(npoints/8):])
            axs[1,2].set_ylabel("notch values")
            axs[1,2].set_xlabel("time")
   
            axs[2,0].plot(tm[int(npoints/8):],plot_dict[key]["delta"][int(npoints/8):])
            axs[2,0].set_ylabel("delta values")
            axs[2,0].set_xlabel("time")
   
            axs[2,1].plot(tm[int(npoints/8):],plot_dict[key]["jagged"][int(npoints/8):])
            axs[2,1].set_ylabel("jagged values")
            axs[2,1].set_xlabel("time")
   
            axs[2,2].plot(tm[int(npoints/8):],plot_dict[key]["nicd"][int(npoints/8):])
            axs[2,2].set_ylabel("nicd values")
            axs[2,2].set_xlabel("time")
   
       	f.suptitle(filename[:-4])
       	plt.subplots_adjust(top=0.9, bottom=0.2, left=0.10, right=0.95, hspace=0.25,wspace=0.5)
       	plt.savefig(input_folder+filename[:-4]+".png",dpi=200)    
       	plt.close()
