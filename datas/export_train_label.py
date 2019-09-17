import csv
import os
import pandas as pd
from pprint import pprint

def grade_mode(list):

    list_set = set(list) 
    frequency_dict = {}
    for i in list_set: 
        frequency_dict[i] = list.count(i)  
    grade_mode = []
    for key, value in frequency_dict.items():  
        if value == max(frequency_dict.values()):
            grade_mode.append(key)
    return grade_mode




dataset = pd.read_csv('faceexp-comparison-data-train-public.csv', header=None,error_bad_lines=False)

new_dataset = {}
names1 = []
names2 = []
names3 = []
types = []
modes = []
for i in range(0,63000):
    name1 = "/home/spark/grocery/FER/codes/FECNet-pytorch/datas/train/" + dataset.iloc[i, 0].split('/')[-1]
    name2 = "/home/spark/grocery/FER/codes/FECNet-pytorch/datas/train/" + dataset.iloc[i, 5].split('/')[-1]
    name3 = "/home/spark/grocery/FER/codes/FECNet-pytorch/datas/train/" + dataset.iloc[i, 10].split('/')[-1]
    if os.path.exists(name1) == False or os.path.exists(name2) \
            == False or os.path.exists(name3) == False:
        continue
    print(name1)
    print(name2)
    print(name3)
    the_type = dataset.iloc[i, 15]
    modes = grade_mode([dataset.iloc[i, 17],dataset.iloc[i, 19],dataset.iloc[i, 21],dataset.iloc[i, 23],dataset.iloc[i, 25],dataset.iloc[i, 27]])
    mode = modes[0]
    print(mode)
    if mode == 1:
        names1.append(name2)
        names2.append(name3)
        names3.append(name1)
    elif mode == 2:
        names1.append(name3)
        names2.append(name1)
        names3.append(name2)
    elif mode == 3:
        names1.append(name1)
        names2.append(name2)
        names3.append(name3)


new_dataset["anchor"] = names1
new_dataset["postive"] = names2
new_dataset["negative"] = names3
# new_dataset[3]=modes
# new_dataset[4]=types
new_data = pd.DataFrame(new_dataset)
pprint(new_data)
print("DataFrame's cols is {}".format(new_data.shape[0]))

new_data.to_csv('pd_triplet_data.csv')
