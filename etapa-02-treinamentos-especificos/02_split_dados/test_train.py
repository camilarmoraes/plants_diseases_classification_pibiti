"""
test_train.py — Divisão Train/Test/Validation com scikit-learn.

Demonstra como dividir o dataset em 3 partições (treino 70%, teste 17%,
validação 10%) usando train_test_split do scikit-learn, e treina uma CNN
com validação cruzada.

Dataset: Fashion MNIST
"""

# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
# Bibliotecas Auxiliares
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)


#Importando e carregando o Fashion Mnist
fashion_mnist = keras.datasets.fashion_mnist


(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

images = np.concatenate((train_images,test_images),axis = 0)
labels = np.concatenate((train_labels,test_labels),axis = 0)

treino,val, treino_rotulo,val_rotulo = train_test_split(images,labels,test_size = 0.10,random_state = 0)

len(treino)
len(val)
len(treino_rotulo)
len(val_rotulo)

treino2,teste, treino_rotulo2,teste_rotulo = train_test_split(treino,treino_rotulo,test_size = 0.166,random_state = 0)

len(treino2)
len(teste)
len(treino_rotulo2)
len(teste_rotulo)

#Colocando o nome dos labels, já que a base de dados possui apenas 1 
class_names = ['Camiseta/Top', 'Calça', 'Sueter', 'Vestido', 'Casaco',
               'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota']


###Pré-Processamento dos dados

#

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

#Importante que os dados de treinamento e de teste sejam pré-processados do mesmo modo
train_images = treino2 / 255.0

test_images = teste / 255.0

'''
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
'''

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
    treino_rotulo2,
    validation_data=(test_images, test_labels),
    epochs=10)

#Salvando o modelo
#model.save("modeloTeste1.h5")


###Avaliando a acurácia
test_loss, test_acc = model.evaluate(test_images,  teste_rotulo, verbose=2)

print('\nTest accuracy:', test_acc)
print('\nTest loss:', test_loss)

loss, acc = model.evaluate(val,  val_rotulo, verbose=2)

print('\nVal accuracy:', acc)
print('\nVal loss:', loss)

###Fazendo predições
predictions = model.predict(val)

print('Predicões: ',predictions)


np.argmax(predictions[0])

test_labels[0]

'''

# Grab an image from the test dataset.
img = test_images[0]

print(img.shape)

# Adiciona a imagem em um batch que possui um só membro.
img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = model.predict(img)

print(predictions_single)
'''



####Tensor Flow Lite
    #Quantizar tornará o modelo menor e potencialmente rodar mais rápido
    




