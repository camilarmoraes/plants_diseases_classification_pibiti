"""
teste_de_camadas.py — CNN com Conv2D para Fashion MNIST.

Primeiro teste de camada convolucional: Conv2D(32) + MaxPool + Dense(128).
Salva o modelo treinado em 'modelo_principal.h5', que depois é reutilizado
pelo script salvando_carregando.py.

Dataset: Fashion MNIST (10 classes de vestuário)
"""

# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras

# Bibliotecas Auxiliares
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


#Importando e carregando o Fashion Mnist
fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


#Colocando o nome dos labels, já que a base de dados possui apenas 1 
class_names = ['Camiseta/Top', 'Calça', 'Sueter', 'Vestido', 'Casaco',
               'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota']



####Explorar o dados antes de testá-los.

#Mostra a existência de 60000 imagens no conjunto de treinamento
train_images.shape

#Existem 60000 labels no conjunto de trinamento
len(train_labels)

#Cada label é um inteiro entre 0 e 9
train_labels

#Mostra a quantidade de imagens no conjunto de teste
test_images.shape

#Conjunto de testes de 10000 labels das imagens
len(test_labels)

###Pré-Processamento dos dados

#
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#Importante que os dados de treinamento e de teste sejam pré-processados do mesmo modo
train_images = train_images / 255.0

test_images = test_images / 255.0

#Mostrando as primeiras 25 imagens do conjunto de treino e mostrar o nome das classes de cada imagem
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


#####Construindo o modelo
        #Requer configurar as camadas e depois compilar os modelos
        
        
###Montando as camdas
model = keras.Sequential([
    tf.keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(3,3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])
#mostrar as estruturas    
 
model.summary()

###Compilando o modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(
    train_images,
    train_labels,
    #validation_data=(test_images, test_labels),
    epochs=10)

#Salvando o modelo
model.save("modelo_principal.h5")


'''
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
# plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.savefig(f'teste9.png')
'''
###Avaliando a acurácia
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)


###Fazendo predições
predictions = model.predict(test_images)

predictions[0]

np.argmax(predictions[0])

test_labels[0]



# Grab an image from the test dataset.
img = test_images[0]

print(img.shape)

# Adiciona a imagem em um batch que possui um só membro.
img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)




####Tensor Flow Lite
    #Quantizar tornará o modelo menor e potencialmente rodar mais rápido
    




