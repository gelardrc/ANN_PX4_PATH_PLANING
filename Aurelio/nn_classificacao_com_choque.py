import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
from keras import models
from keras import layers
import tensorflow as tf

dataset = pd.read_csv("/home/gelo/src/ANN_PX4_PATH_PLANING/Aurelio/dataset_classificador_com_choque_01.csv")

print(dataset.shape) ## 30 0000 // 18

dataset_treino = dataset.iloc[0:10000,0:18]
dataset_validacao = dataset.iloc[10000:20000,0:18]
dataset_teste = dataset.iloc[20000:30000,0:18]

x_train = dataset_treino.iloc[:,0:12]
y_train = dataset_treino.iloc[:,12:18]
x_val = dataset_validacao.iloc[:,0:12]
y_val = dataset_validacao.iloc[:,12:18]
x_teste = dataset_teste.iloc[:,0:12]
y_teste = dataset_teste.iloc[:,12:18]
print(x_teste.shape) 
print(y_teste)
callback = tf.keras.callbacks.EarlyStopping(monitor='accuracy', patience=5)
model = models.Sequential()
model.add(layers.Dense(16,activation='sigmoid',input_shape=(x_train.shape[1],)))
#model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(6,activation='softmax'))
model.compile(optimizer = 'Ftrl',
                loss='categorical_crossentropy',
                metrics = ['accuracy']
                )

history = model.fit(x_train,
          y_train,
          epochs = 500,
          #batch_size = 500,
          validation_data = (x_val, y_val), 
          callbacks = [callback])

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

score =  model.evaluate(x_teste,y_teste)
print(score)