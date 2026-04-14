"""
salvando_carregando.py — Teste de persistência de modelo.

Carrega um modelo treinado ('modelo_principal.h5') e avalia com dados
de teste divididos usando train_test_split, demonstrando salvar/carregar
modelos Keras e verificar a reprodutibilidade.

Dataset: Fashion MNIST
"""

import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

novo_modelo = tf.keras.models.load_model('modelo_principal.h5')

novo_modelo.summary()

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#Colocando o nome dos labels, já que a base de dados possui apenas 1 
class_names = ['Camiseta/Top', 'Calça', 'Sueter', 'Vestido', 'Casaco',
               'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota']

#Fazer split dos testes para ver a aleatoriedade.
imagens_treino,imagens_test,rotulos_treino,rotulos_teste = train_test_split(test_images,test_labels,test_size = 0.2,random_state=0)

imagens_treino2,imagens_test2,rotulos_treino2,rotulos_teste2 = train_test_split(test_images,test_labels,test_size = 0.2,random_state=0, shuffle = True)



loss, acc = novo_modelo.evaluate(imagens_treino,rotulos_treino, verbose=2)
loss2, acc2 = novo_modelo.evaluate(imagens_treino2,rotulos_treino2, verbose=2)
print('Novo modelo, accuracy: {:5.2f}%'.format(100 * acc))

#Carregar todos os modelos salvos
#Para cada modelo, utilizar o evaluate com o mesmo dataset
#Para cada evaluate, criar um arquivo txt com o resultado da acurácia
