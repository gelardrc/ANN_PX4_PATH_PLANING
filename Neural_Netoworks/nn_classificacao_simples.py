import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
#from pandas import mean
from keras import models
from keras import layers
import tensorflow as tf
from cross_validation import kfold
#from cross_validation import test

######## aparentemente salva a melhor rede ########

def get_model_name(k):
    return 'model_'+str(k)+'.h5'




fig = plt.figure(dpi=200)


#### Training loss ######
ax = fig.add_subplot(2, 2, 1)
ax.set_xlabel('Epochs')
ax.set_ylabel('Training Loss')
#ax.set_title('Training and validation loss')

#### Validation loss ######
ax1 = fig.add_subplot(2, 2, 2)
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Validation Loss')

#### Acuracia loss ########

ax2 = fig.add_subplot(2, 2, 3)
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')

#### Acuracia validation_loss ########

ax3 = fig.add_subplot(2, 2, 4)
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Accuracy validation')

######### teste error #############
#ax4 = fig.add_subplot(1,1,2)
# ax4.set_xlabel('Epochs')
#ax4.set_ylabel('Test error')
#ax2.set_title('Training and validation acc')

dataset = pd.read_csv(
    "/home/gelo/codes/ANN_PX4_PATH_PLANING/DATASETS/dirsjtk_3d.csv")

# tentativa de scale
##dataset.iloc[:,0:12] = (dataset.iloc[:,0:12] - dataset.iloc[:,0:12].mean())/(dataset.iloc[:,0:12].max()-dataset.iloc[:,0:12].min())
#
# Bagunça o dataset
#dataset = dataset.sample(frac=1)
#
#### divide entre treino e teste ###
#dataset_treino = dataset.iloc[0:120000,0:18]
#
#dataset_teste = dataset.iloc[120000:180000,0:18]
####################################
#
########## divide o teste em x_teste e y_ teste ########
#x_teste = dataset_teste.iloc[:,0:12]
#
#y_teste = dataset_teste.iloc[:,12:18]
########################################################
#
#
####### define Y como dataset_treino ###########
#
#Y = dataset_treino.iloc[0:120000,12:18]
#X = dataset_treino.iloc[0:120000,0:12]
#
######## Numero de kfold ##########
#kf = KFold(n_splits = 5)
###################################
#
############ define a função kfold #######
#skf = StratifiedKFold(n_split = 5, random_state = 7, shuffle = True)
##########################################
#
#
#
#dataset_treino = dataset.iloc[0:60000, 0:18]
#dataset_validacao = dataset.iloc[60000:120000, 0:18]
#dataset_teste = dataset.iloc[120000:180000, 0:18]

#x_train = dataset_treino.iloc[:, 0:12]
#y_train = dataset_treino.iloc[:, 12:18]
#x_val = dataset_validacao.iloc[:, 0:12]
#y_val = dataset_validacao.iloc[:, 12:18]
#x_teste = dataset_teste.iloc[:, 0:12]
#y_teste = dataset_teste.iloc[:, 12:18]

callback = tf.keras.callbacks.EarlyStopping( monitor='mse', patience=5)

# for train_index, val_index in kf.split(X,Y):

# tf.keras.optimizers.RMSprop(
#    learning_rate=5,
#    name="RMSprop"
# )

optimizer = ["SGD", "RMSprop", "Adam", "Adadelta","Adagrad", "Adamax", "Nadam", "Ftrl"]

#modelos = [[32,64,128],[18,32,64],[9,18,128],[18,18,18],[32,128,64],[16,128,64],[18,24,62],[64,16,8],[16,128,64]]

modelos = [[81,212,29]]
#modelos = [[64,16,8]]#,[32,8,8],[16,4,8]]
graph = [0,0,0,0,0,0,0,0]

input_labels = ['h','g','f','alvo_x','alvo_y','alvo_z','choque0','choque1','choque2','choque3','choque4','choque5']
output_labels = ['acao0','acao1','acao2','acao3','acao4','acao5']

score= []
legend = []
n_fold = 10

for opt in optimizer:
    for i in range(1,n_fold):
        print(i)
        data_train,data_test = kfold(dataset,i)
        #n_layer = opt[0] 
        x_train = data_train[input_labels]
        x_teste = data_test[input_labels]
        y_train = data_train[output_labels]
        y_teste = data_test[output_labels] 
        model = models.Sequential()
        model.add(layers.Dense(89, activation='relu',
                  input_shape=(x_train.shape[1],)))
        #model.add(layers.Dense(212, activation='relu'))
        #model.add(layers.Dense(29, activation='relu'))
        model.add(layers.Dense(6, activation='softmax'))
        model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics='mse'#[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.5)]
                      )

        history = model.fit(x_train,
                            y_train,
                            epochs=8000,
                            batch_size=500,
                            #validation_data=(x_val, y_val),
                            callbacks=[callback],
                            verbose=0)

        score.append(model.evaluate(x_teste,y_teste))                    
        loss = history.history['loss']
        epochs = range(1, len(loss) + 1)
        line, = ax.plot(epochs, history.history['loss'])
        # ax.legend([opt[0]])
        #line1, = ax1.plot(epochs, history.history['val_loss'])
        # ax1.legend([opt[0]])
        line2, = ax2.plot(epochs, history.history['mse'])
        # ax2.legend([opt[0]])
        #line3, = ax3.plot(epochs, history.history['val_mse'])
        # ax3.legend([opt[0]])

        legend.append(line)

        # history.history['binary_accuracy']
        # history.history['val_binary_accuracy']
        #model.evaluate(x_teste, y_teste)
        #epochs = range(1, len(loss) + 1)
        #plt.plot(epochs, loss, 'ro', label='Training loss')
        #plt.plot(epochs, val_loss, 'b', label='Validation loss')
        #plt.title('Training and validation loss')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()

        # plt.show()
        # plt.clf()

        #plt.plot(epochs, acc, 'ro', label='Training acc')
        #plt.plot(epochs, val_acc, 'b', label='Validation acc')
        #plt.title('Training and validation acc')
        # plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        # plt.legend()
        # plt.show()

    #
# ax4.legend([line1,line2,line3,line4,line5,line6],["SGD", "RMSprop", "Adam", "Adadelta","Adagrad", "Adamax", "Nadam", "Ftrl"]

labels = ['SGD', 'RMSprop', 'Adam', 'Adadelta','Adagrad', 'Adamax', 'Nadam', 'Ftrl']
fig.legend(legend, labels, 'lower left')


#ax.legend([[32,64,128],[18,32,64],[9,18,128],[18,18,18],[32,128,64],[16,128,64],[18,24,62],[64,16,8]])
#ax1.legend([[32,64,128],[18,32,64],[9,18,128],[18,18,18],[32,128,64],[16,128,64],[18,24,62],[64,16,8]])
#ax2.legend([[32,64,128],[18,32,64],[9,18,128],[18,18,18],[32,128,64],[16,128,64],[18,24,62],[64,16,8]])
#ax3.legend([[32,64,128],[18,32,64],[9,18,128],[18,18,18],[32,128,64],[16,128,64],[18,24,62],[64,16,8]])




print(score)
plt.show()


# model.save("/home/gelo/codes/ANN_PX4_PATH_PLANING/Redes_salvas/dritk_banco.h5")
