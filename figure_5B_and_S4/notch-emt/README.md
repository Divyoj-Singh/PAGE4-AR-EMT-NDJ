Codes written by Marcelo Boareto (marceloboareto at gmail)

####Paper reference:

http://rsif.royalsocietypublishing.org/content/13/118/20151106

## Notch-Jagged signalling can give rise to clusters of cells exhibiting a hybrid epithelial/mesenchymal phenotype

######Marcelo Boareto, Mohit Kumar Jolly, Aaron Goldman, Mika Pietilä, Sendurai A. Mani, Shiladitya Sengupta, Eshel Ben-Jacob, Herbert Levine, Jose’ N. Onuchic




1 Cell simullations:

https://github.com/mboareto/Notch_EMT_JRSocInterface2016/blob/master/EMTNotch_1cell.ipynb



Tissue simulations:

https://github.com/mboareto/Notch_EMT_JRSocInterface2016/blob/master/NotchEMT_2Dsimulations.ipynb



auxilary functions:

https://github.com/mboareto/Notch_EMT_JRSocInterface2016/blob/master/auxilary_functions.py





#### Erratum (Jan 2017)
I realized a minor error in the code, in the definitions of the auxiliary functions Py and Pl (I thank Federico Bocci
for pointing this out). I had a for loop up to n, while the correct is to n+1 (see lines 112-124 of auxilary_function.py). 
This error does not affect the 1 Cell simulations, but small quantitative differences are observed in the tissue level simulations.
The interpretation and conclusions of the model remains the same. The corrected version is presented at: 

https://github.com/mboareto/Notch_EMT_JRSocInterface2016/blob/master/NotchEMT_2Dsimulations_corrected.ipynb
