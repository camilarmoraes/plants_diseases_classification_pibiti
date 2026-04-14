"""
programaTFLite.py — Pipeline completo: avaliação, conversão TFLite e inferência.

Avalia 10 modelos treinados, converte o melhor para TFLite e TFLite
quantizado, e compara as predições dos três formatos (TF, TFLite,
TFLite Quantizado).

Dataset: Fashion MNIST
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# ─── Configuração ─────────────────────────────────────────────────────────────
NUM_MODELOS = 10
MODELS_DIR = './'
BEST_MODEL_FILE = 'modelo8.h5'
TFLITE_OUTPUT = 'modelo_tflite.tflite'
TFLITE_QUANT_OUTPUT = 'modelo_tflite_quant.tflite'
RESULTS_FILE = 'arquivo_resultado.txt'
PREDICTIONS_TF_FILE = 'predicoesTensorFlow.txt'
PREDICTIONS_TFLITE_FILE = 'predicoesTFLite.txt'
# ──────────────────────────────────────────────────────────────────────────────

# Carregamento do dataset
fashion_mnist = keras.datasets.fashion_mnist
(input_train, target_train), (input_test, target_test) = fashion_mnist.load_data()

class_names = ['Camiseta/Top', 'Calça', 'Sueter', 'Vestido', 'Casaco',
               'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota']

# Avaliação de todos os modelos
modelos = [None] * NUM_MODELOS
for i in range(NUM_MODELOS):
    modelos[i] = tf.keras.models.load_model(f'{MODELS_DIR}modelo{i+1}.h5')
    loss, acc = modelos[i].evaluate(input_test, target_test, verbose=0)
    with open(RESULTS_FILE, "a") as arquivo:
        arquivo.write(f"\n Modelo{i+1} = Loss: {loss}  Acc: {acc} \n")

# Predições com o melhor modelo TensorFlow
model = tf.keras.models.load_model(f'{MODELS_DIR}{BEST_MODEL_FILE}')
predictions = model.predict(input_test)

for i in range(10):
    erros = 0
    acertos = 0
    for x in range(len(input_test)):
        if target_test[x] == i:
            if np.argmax(predictions[x]) == target_test[x]:
                acertos += 1
            else:
                erros += 1
    total = acertos + erros
    pct = (acertos / total) * 100 if total > 0 else 0
    with open(PREDICTIONS_TF_FILE, "a") as pred:
        pred.write(f'Classe {i}: acertos={acertos}, erros={erros} '
                   f'— Porcentagem={pct:.2f}%\n')

# ─── Conversão para TFLite ───────────────────────────────────────────────────
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflmodel = converter.convert()
with open(TFLITE_OUTPUT, 'wb') as f:
    f.write(tflmodel)

# ─── Conversão para TFLite Quantizado ────────────────────────────────────────
converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant = converter_quant.convert()
with open(TFLITE_QUANT_OUTPUT, 'wb') as f:
    f.write(tflite_quant)

# ─── Inferência TFLite ───────────────────────────────────────────────────────
interpreter = tf.lite.Interpreter(TFLITE_OUTPUT)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Redimensiona para processar todo o conjunto de teste
interpreter.resize_tensor_input(
    input_details[0]['index'], (len(input_test), 28, 28, 1))
interpreter.resize_tensor_input(
    output_details[0]['index'], (len(input_test), 10))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_data = np.expand_dims(input_test, axis=3)
input_data = np.float32(input_data)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

model_predictions = interpreter.get_tensor(output_details[0]['index'])
print("Prediction results shape:", model_predictions.shape)

# Comparação TFLite por classe
for i in range(10):
    erros = 0
    acertos = 0
    for x in range(len(input_test)):
        if target_test[x] == i:
            if np.argmax(model_predictions[x]) == target_test[x]:
                acertos += 1
            else:
                erros += 1
    total = acertos + erros
    pct = (acertos / total) * 100 if total > 0 else 0
    with open(PREDICTIONS_TFLITE_FILE, "a") as pred:
        pred.write(f'Classe {i}: acertos={acertos}, erros={erros} '
                   f'— Porcentagem={pct:.2f}%\n')
