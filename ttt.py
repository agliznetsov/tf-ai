import numpy as np
import tensorflow as tf
from plot_history import *

data = np.genfromtxt('./data/t.csv', delimiter=',')
np.random.shuffle(data)
m = data.shape[0]
m_train = int(m * 0.5)
data = np.split(data, [9], axis=1)
X = np.split(data[0], [m_train])
Y = np.split(data[1], [m_train])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(9, kernel_regularizer=tf.keras.regularizers.l2(0.1), activation=tf.nn.relu),
    # tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(9, kernel_regularizer=tf.keras.regularizers.l2(0.1), activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X[0], Y[0], epochs=500, verbose=0)
res = model.evaluate(X[1], Y[1], verbose=0)
print('Test evaluation: ' + str(res))

plot_history([('baseline', history)])
