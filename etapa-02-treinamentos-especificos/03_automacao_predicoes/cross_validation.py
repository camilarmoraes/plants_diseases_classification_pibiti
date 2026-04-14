"""
cross_validation.py — K-Fold Cross Validation com CNN.

Aplica K-Fold Cross Validation (10 folds) no dataset Fashion MNIST
usando uma CNN com Conv2D + Dense. Calcula acurácia e loss por fold
e apresenta a média e desvio padrão.

Dataset: Fashion MNIST
"""

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import KFold
import numpy as np

# ─── Configuração ─────────────────────────────────────────────────────────────
MODEL_SAVE_DIR = './modelos/'           # Diretório para salvar modelos por fold
img_width, img_height, img_num_channels = 28, 28, 1
no_classes = 10
verbosity = 1
num_folds = 10 #folds = subsets que serão divididos o dataset bruto

fashion_mnist = keras.datasets.fashion_mnist

(input_train, target_train), (input_test, target_test) = fashion_mnist.load_data()

class_names = ['Camiseta/Top', 'Calça', 'Sueter', 'Vestido', 'Casaco',
               'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota']


# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')

# Normalize data
input_train = input_train / 255
input_test = input_test / 255

# Define per-fold score containers
acc_per_fold = []
loss_per_fold = []

# Merge inputs and targets
inputs = np.concatenate((input_train, input_test), axis=0)
targets = np.concatenate((target_train, target_test), axis=0)

#inputs = shuffle(inputs)


# Define the K-fold Cross Validator
kfold = KFold(n_splits=10, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
save_model = 1
for train, test in kfold.split(inputs, targets): #generate indices to split data into training and test set
    model = keras.Sequential([
            tf.keras.layers.Conv2D(32,(2,2),activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(3,3),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')])
    
    model.compile(loss = 'sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])



  # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Treinamento para o subset {fold_no} ...')

  # Fit data to model
    history = model.fit(inputs[train], targets[train],  
              #batch_size=batch_size,
              epochs=10,
              verbose=2)
#Salvando o modelo
    # model.save(f'{MODEL_SAVE_DIR}modelo{save_model}.h5')
  # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Pontuação por subset {fold_no}: {model.metrics_names[0]} do {scores[0]}; {model.metrics_names[1]} do {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    loss_per_fold.append(scores[0])

  # Increase fold number
    save_model = save_model + 1
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Pontuação por subset:')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Subset {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Média da pontuação de todos os subsets:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


