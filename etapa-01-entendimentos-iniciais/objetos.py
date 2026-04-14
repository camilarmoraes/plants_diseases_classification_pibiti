"""
objetos.py — Classificação de objetos escolares com TFLite Model Maker.

Utiliza o TFLite Model Maker para treinar um classificador de imagens
com 6 classes de objetos do dia a dia (borracha, caneta, clip, grafite,
lápis, post-it), capturadas manualmente.

Pipeline:
    1. Carrega imagens do diretório organizado por classes
    2. Divide em treino (80%), validação (10%) e teste (10%)
    3. Treina com o Model Maker (transfer learning)
    4. Avalia e visualiza predições
    5. Exporta para TFLite e TFLite quantizado (float16)
    6. Testa com diferentes arquiteturas (MobileNet V2, Inception V3)
"""

import os
import numpy as np
import tensorflow as tf
assert tf.__version__.startswith('2')

from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader
import matplotlib.pyplot as plt

# ─── Configuração ─────────────────────────────────────────────────────────────
# Diretório contendo as imagens organizadas em subpastas por classe.
# Estrutura esperada:
#   IMAGE_DIR/
#   ├── Borracha/
#   ├── Caneta/
#   ├── Clip/
#   ├── Grafite/
#   ├── Lapis/
#   └── Post-it/
IMAGE_DIR = './dataset_objetos'
EXPORT_DIR = './output'
EPOCHS_INICIAL = 10
EPOCHS_DETALHADO = 50
EPOCHS_FINAL = 40
# ──────────────────────────────────────────────────────────────────────────────


# Carregando os dados de entrada (separa automaticamente por rótulos/subpastas)
data = DataLoader.from_folder(IMAGE_DIR)  # Suporta apenas JPEG e PNG

# Primeiro teste rápido: split 70/30, treinamento por 10 epochs
train_data, test_data = data.split(0.7)
model = image_classifier.create(train_data, epochs=EPOCHS_INICIAL)
loss, accuracy = model.evaluate(test_data)
model.export(export_dir=EXPORT_DIR)


# Detalhando o processo: split 80/10/10
data = DataLoader.from_folder(IMAGE_DIR)
train_data, rest_data = data.split(0.8)
validation_data, test_data = rest_data.split(0.5)

# Visualizando exemplos de imagens do dataset
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(data.gen_dataset().unbatch().take(25)):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image.numpy(), cmap=plt.cm.gray)
    plt.xlabel(data.index_to_label[label.numpy()])
plt.show()

# Treinamento com validação — a partir de 26 epochs já é satisfatório
model = image_classifier.create(
    train_data, validation_data=validation_data, epochs=EPOCHS_DETALHADO)
model.summary()
loss, accuracy = model.evaluate(test_data)


# Visualização das predições (azul = correto, vermelho = incorreto)
def get_label_color(val1, val2):
    return 'blue' if val1 == val2 else 'red'

plt.figure(figsize=(20, 20))
predicts = model.predict_top_k(test_data)
for i, (image, label) in enumerate(test_data.gen_dataset().unbatch().take(100)):
    ax = plt.subplot(10, 10, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image.numpy(), cmap=plt.cm.gray)
    predict_label = predicts[i][0][0]
    color = get_label_color(predict_label, test_data.index_to_label[label.numpy()])
    ax.xaxis.label.set_color(color)
    plt.xlabel('Predicted: %s' % predict_label)
plt.show()


# Exportação para TFLite
model.export(export_dir=EXPORT_DIR)
model.export(export_dir=EXPORT_DIR, export_format=ExportFormat.LABEL)
model.evaluate_tflite('model.tflite', test_data)

# Exportação com quantização float16
config = QuantizationConfig.for_float16()
model.export(export_dir=EXPORT_DIR, tflite_filename='model_fp16.tflite',
             quantization_config=config)

# Teste com MobileNet V2
model = image_classifier.create(
    train_data,
    model_spec=model_spec.get('mobilenet_v2'),
    validation_data=validation_data,
    epochs=100)
loss, accuracy = model.evaluate(test_data)

# Teste com Inception V3
inception_v3_spec = image_classifier.ModelSpec(
    uri='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1')
inception_v3_spec.input_image_shape = [299, 299]

model = image_classifier.create(
    train_data, validation_data=validation_data, epochs=EPOCHS_FINAL)
loss, accuracy = model.evaluate(test_data)
