import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
from keras import models
from keras import layers
from keras.regularizers import l2
import tensorflow as tf

dataset = pd.read_csv("/home/gelo/codes/ANN_PX4_PATH_PLANING/DATASETS/dirsjtk_3d.csv")


dataset = dataset.sample(frac=1)


print(dataset.shape) ## 30 0000 // 18

dataset_treino = dataset.iloc[0:60000,0:18]
dataset_validacao = dataset.iloc[60000:120000,0:18]
dataset_teste = dataset.iloc[120000:180000,0:18]

x_train = dataset_treino.iloc[:,0:12]
y_train = dataset_treino.iloc[:,12:18]
x_val = dataset_validacao.iloc[:,0:12]
y_val = dataset_validacao.iloc[:,12:18]
x_teste = dataset_teste.iloc[:,0:12]
y_teste = dataset_teste.iloc[:,12:18]
print(x_teste.shape) 
print(y_teste)
print(x_train)
callback = tf.keras.callbacks.EarlyStopping(monitor='binary_accuracy', patience=5)


tf.keras.optimizers.RMSprop(
    learning_rate=0.01,
    name="RMSprop"
)

model = models.Sequential()
model.add(layers.Dense(16,kernel_regularizer=l2(0.01),activation='relu',input_shape=(x_train.shape[1],)))
model.add(layers.Dense(128,activation='relu'))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(6,activation='softmax'))
model.compile(optimizer = 'RMSprop',
                loss='categorical_crossentropy',
                metrics = [tf.keras.metrics.BinaryAccuracy( name="binary_accuracy", dtype=None, threshold=0.7)]
                )


learning = [0,0.001,0.003,0.01,0.02,0.08,0.1,0.2,0,5]

rate = []
score =[]
for learn in learning:
    
    tf.keras.optimizers.RMSprop(
    learning_rate=learn,
    name="RMSprop"
)    
    history = model.fit(x_train,
              y_train,
              epochs = 200,
              batch_size = 500,
              validation_data = (x_val, y_val), 
              callbacks = [callback],
              verbose=0)
    
    score.append(model.evaluate(x_teste,y_teste)) #=  model.evaluate(x_teste,y_teste)
    rate.append(learn)

#model.save("/home/gelo/codes/ANN_PX4_PATH_PLANING/Redes_salvas/dritk_banco.h5")


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
acc = history.history['binary_accuracy']
val_acc = history.history['val_binary_accuracy']
plt.plot(epochs, acc, 'ro', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
    

for i in range(len(learning)):
    print('score -->');print(score[i])
    print('rate -->');print(rate[i]);print("\n")


#print(score)

plt.show()
