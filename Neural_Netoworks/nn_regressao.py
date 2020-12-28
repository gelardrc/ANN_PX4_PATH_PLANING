import numpy as np
import pandas as pd
from keras import models
from keras import layers
import matplotlib.pyplot as plt

dataset = pd.read_csv("dataset_classificador.csv")
dataset_treino = dataset.iloc[0:1200,0:9]
dataset_validacao = dataset.iloc[1200:2013,0:9]
x_train = dataset_treino.iloc[:,0:8]
y_train = dataset_treino.iloc[:,8]
x_val = dataset_validacao.iloc[:,0:8]
y_val = dataset_validacao.iloc[:,8]
#dataset com (2013,9)
print(x_train.shape) 
print(y_train)

model = models.Sequential()
model.add(layers.Dense(32,activation='relu',input_shape=(8,)))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer = 'adam',
                loss = 'mse',
                metrics = ['mae'])
history = model.fit(x_train,
          y_train,
          epochs = 200,
          #batch_size = 500,
          validation_data = (x_val, y_val) )


loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

plt.clf()

mae = history.history['mae']
val_mae = history.history['val_mae']
plt.plot(epochs, mae, 'bo', label='Training mae')
plt.plot(epochs, val_mae, 'b', label='Validation mae')
plt.title('Training and validation mae')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

