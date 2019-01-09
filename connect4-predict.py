import tensorflow as tf
from plot_history import *

labels = {
    "b": 0,
    "x": 1,
    "o": 2,
    "loss": 0,
    "draw": 1,
    "win": 2
}


def parse(s):
    return float(labels.get(s))


converters = {}
for i in range(0, 43):
    converters[i] = parse

data = np.genfromtxt('./data/connect4-predict.csv', delimiter=',', dtype=float, converters=converters, encoding='latin1') # converters=converters,

# np.save('./data/connect4-predict.npy', data)
# data = np.load('./data/connect4-predict.npy')

np.random.shuffle(data)

data = np.split(data, [42], axis=1)
X = data[0]
Y = data[1]
# print(X.shape)
# print(Y.shape)
# print(X[0])
# print(Y[0])
# tmp = np.flip(X[0].reshape((7, 6)).T)
# print(tmp)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(150, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    # tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(150, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    # tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(X, Y, validation_split=0.2, epochs=15, verbose=1)
# res = model.evaluate(X[1], Y[1], verbose=0)
# print('Test accuracy: ' + str(res[1]))

plot_history([('baseline', history)], [['loss', 'val_loss'], ['acc', 'val_acc']])
