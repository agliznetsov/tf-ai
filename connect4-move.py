import numpy as np
import tensorflow as tf
from plot_history import *

data = np.genfromtxt('./data/connect4-move.data', delimiter=',')
np.random.shuffle(data)
m = data.shape[0]
data = np.split(data, [42], axis=1)
X = data[0]
# X = X * -1 + 1
Y = data[1]
print(X.shape)
print(Y.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0),
    tf.keras.layers.Dense(7, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X, Y, validation_split=0.2, epochs=50, verbose=1)
# res = model.evaluate(X[1], Y[1], verbose=0)
# print('Test accuracy: ' + str(res[1]))

plot_history([('baseline', history)], [['loss', 'val_loss'], ['acc', 'val_acc']])
