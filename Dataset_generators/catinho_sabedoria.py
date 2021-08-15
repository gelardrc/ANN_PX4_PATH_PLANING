import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from random import randint
import pandas as pd
from keras import models
from keras import layers
import tensorflow as tf



def random_model(ax, ax1, ax2, ax3):
    #### lista de otimizadores ########

    optimizer = ['SGD', 'RMSprop', 'Adam', 'Adadelta',
                 'Adagrad', 'Adamax', 'Nadam', 'Ftrl']

    opt = rnd.choice(optimizer)

    ### função callback ######

    callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)

    #### batch size ##
    batch_size1 = [i for i in range(100, 1000, 50)]
    batch_size = rnd.choice(batch_size1)
    
    #### number of layers ###

    n_layers = randint(0, 5)

    ### number of neurons ##
    best =  False
    n_neurons = [randint(1, 256) for i in range(n_layers+1)]

    model = models.Sequential()

    ### essa primeira camada sempre ira existir porém com o numero de neuronios aleatório ########

    model.add(layers.Dense(n_neurons[0], activation='relu',
              input_shape=(x_train.shape[1],)))

    ######### adiciona o numero de camadas ocultas e set os neuronios para um numero aleário entre 1,256 ########
    for i in range(n_layers-1):
        model.add(layers.Dense(n_neurons[i+1], activation='relu'))

    ####### essa ultima camada é fixa #########
    model.add(layers.Dense(6, activation='softmax'))

    # Compila modelo #########3
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',
                  # [tf.keras.metrics.BinaryAccuracy(name="binary_accuracy", dtype=None, threshold=0.8)]
                  metrics='accuracy'
                  )

    #### FAZ O FIT ###########
    history = model.fit(x_train,
                        y_train,
                        epochs=1000,
                        batch_size=batch_size,
                        validation_data=(x_val, y_val),
                        callbacks=[callback],
                        verbose=0)

    score = model.evaluate(x_teste, y_teste)
    loss = history.history['loss']
    epochs = range(1, len(loss) + 1)
    

    line, = ax.plot(epochs, history.history['loss'])
    
    # ax.legend([opt[0]])
    
    line1, = ax1.plot(epochs, history.history['val_loss'])
    
    # ax1.legend([opt[0]])
    
    line2, = ax2.plot(epochs, history.history['accuracy'])
    
    # ax2.legend([opt[0]])
    
    line3, = ax3.plot(epochs, history.history['val_accuracy'])
    
    
    # ax3.legend([opt[0]])

    return score, n_layers, n_neurons, opt, batch_size, line


def banco():

    dataset = pd.read_csv(
        "/home/gelo/codes/ANN_PX4_PATH_PLANING/DATASETS/dirsjtk_3d.csv")

    dataset_treino = dataset.iloc[0:60000, 0:18]
    dataset_validacao = dataset.iloc[60000:120000, 0:18]
    dataset_teste = dataset.iloc[120000:180000, 0:18]

    x_train = dataset_treino.iloc[:, 0:12]
    y_train = dataset_treino.iloc[:, 12:18]
    x_val = dataset_validacao.iloc[:, 0:12]
    y_val = dataset_validacao.iloc[:, 12:18]
    x_teste = dataset_teste.iloc[:, 0:12]
    y_teste = dataset_teste.iloc[:, 12:18]

    return x_train, y_train, x_val, y_val, x_teste, y_teste


def draw():
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

    return ax, ax1, ax2, ax3, fig


ax, ax1, ax2, ax3, fig = draw()

score = []

x_train, y_train, x_val, y_val, x_teste, y_teste = banco()

###### callback para treino #######
legends = []
labels = []
melhor = [0,0]

for f in range(30):
    #print('estou aqui -->', f)
    pontuação, n_layers, n_neurons, opt, batch_size, line= random_model(ax, ax1, ax2, ax3)
    
    legends.append(line)
    #legendary = legenda()
    score.append(pontuação)
    labels.append(["Nºlayers:", n_layers+1, "Nºneurons:",n_neurons, opt, "Batch size :",batch_size])
    print(score)
    

print(labels)

list_line = list(legends)

fig.legend(list_line, labels, 'lower left')

#plt.show()


#fig= plt.figure()
#ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('Epochs')
#ax.set_ylabel('Training Loss')
#
#x = np.linspace(0.0, 2.0)
# y = for f in sin(x)
#teste = ["sgd",'rms']
#i, = ax.plot(x,x*x)
#line2, = ax.plot(x,x*x)
# ax.legend([teste[0],teste[1]])
#
# plt.show()
#
#
#
#coco = [0,1,2,3]
#bunda = [[0,1,2,3,4],[4,3,2,1],[6,7,8,9],[4,5,6,8]]
#it = np.zeros([1,4])
#
#
#print (it)
#
#
# for i in zip(coco,bunda):
#    print(i)
#    num = i[1]
#    num[0]= 999999999
#    print(i)
