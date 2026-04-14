"""
testando_mnist.py — Teste rápido de classificação MNIST com Dropout.

Modelo Sequential com Dropout para classificação de dígitos. Inclui
plot de accuracy/val_accuracy e salvamento do gráfico.

Dataset: MNIST
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


#model.fit(x_train, y_train, epochs=5)
history = model.fit(
    x_train,
    y_train,
    validation_data=(x_test, y_test),
    epochs=5)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig('grafico.png')

loss, acc = model.evaluate(x_test,  y_test, verbose=2)
print("loss: ",loss)
print("acc: ", acc)