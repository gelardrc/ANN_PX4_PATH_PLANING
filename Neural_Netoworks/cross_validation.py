import numpy as np
import pandas as pd

def kfold(dataset,fold):

    baby_legs = round(dataset.shape[0]/10)
    data_test =dataset
    data_train = dataset
    
    if fold == 1:
        data_test = dataset.iloc[0:baby_legs,:]
        data_train = dataset.iloc[baby_legs:dataset.shape[0],:]
        
        

    if fold == 2:
        
        data_test = dataset.iloc[baby_legs:2*baby_legs,:]
        
        data_train = dataset.iloc[0:baby_legs,:] 
        
        data_train.append(dataset.iloc[2*baby_legs:dataset.shape[0],:])

    if fold == 3: 
        
        data_test = dataset.iloc[2*baby_legs:3*baby_legs,:]
        
        data_train = dataset.iloc[0:2*baby_legs,:] 
        
        data_train.append(dataset.iloc[3*baby_legs:dataset.shape[0],:])

    if fold == 4:
        data_test = dataset.iloc[3*baby_legs:4*baby_legs,:]
        data_train = dataset.iloc[0:3*baby_legs,:] 
        data_train.append(dataset.iloc[4*baby_legs:dataset.shape[0],:])

    if fold == 5:
        data_test = dataset.iloc[4*baby_legs:5*baby_legs,:]
        data_train = dataset.iloc[0:4*baby_legs,:]
        data_train.append(dataset.iloc[5*baby_legs:dataset.shape[0],:] )

    if fold == 6:
        data_test = dataset.iloc[5*baby_legs:6*baby_legs,:]
        data_train = dataset.iloc[0:5*baby_legs,:]
        data_train.append(dataset.iloc[6*baby_legs:dataset.shape[0],:] )

    if fold == 7:
        data_test = dataset.iloc[6*baby_legs:7*baby_legs,:]
        data_train = dataset.iloc[0:6*baby_legs,:]  
        data_train.append(dataset.iloc[7*baby_legs:dataset.shape[0],:])

    if fold == 8:
        data_test = dataset.iloc[7*baby_legs:8*baby_legs,:]
        data_train = dataset.iloc[0:7*baby_legs,:]  
        data_train.append(dataset.iloc[8*baby_legs:dataset.shape[0],:])


    if fold == 9:
        data_test = dataset.iloc[8*baby_legs:9*baby_legs,:]
        data_train = dataset.iloc[0:8*baby_legs,:]  
        data_train.append(dataset.iloc[9*baby_legs:dataset.shape[0],:] )


    if fold == 10:
        data_test = dataset.iloc[9*baby_legs:10*baby_legs,:]
        data_train = dataset.iloc[0:9*baby_legs,:] 
        #data_train.append(dataset.iloc[2*baby_legs:dataset.shape[0],:])

       
    return data_train,data_test


#dataset = pd.read_csv("/home/gelo/codes/ANN_PX4_PATH_PLANING/DATASETS/dirsjtk_3d.csv")
#
#for i in range(1,10):
#    treino, teste = kfold(dataset,i)
# 
#    print(teste)



