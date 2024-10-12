import os
import numpy as np
import math
import random
from os import listdir
random.seed(77)
from os.path import isfile, join

#Read file names
path = "D:/BaiduNetdiskDownload/WHU-CD/label/"
list = sorted(os.listdir(path))
list = random.sample(list, len(list))


N_total = len(list)
N_train = math.floor(N_total*0.8)
N_val   = math.floor((N_total-N_train)/2)
N_test  = N_total - N_train - N_val
print(f'Total images = {N_total}, Train/Val/Test = {N_train}/{N_val}/{N_test}.')

list_train = list[0:N_train]
list_val = list[N_train:N_train+N_val]
list_test = list[N_train+N_val:N_total]
print(f'Total images = {N_total}, Train/Val/Test = {len(list_train)}/{len(list_val)}/{len(list_test)}.')

textfile = open("train.txt", "w")
for element in list_train:
    textfile.write(element + "\n")
textfile.close()

textfile = open("val.txt", "w")
for element in list_val:
    textfile.write(element + "\n")
textfile.close()

textfile = open("test.txt", "w")
for element in list_test:
    textfile.write(element + "\n")
textfile.close()

    
    
    

   

