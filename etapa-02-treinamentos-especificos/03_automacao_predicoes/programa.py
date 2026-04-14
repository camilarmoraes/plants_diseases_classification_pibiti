"""
programa.py — Avaliação de múltiplos modelos e inferência TFLite.

Carrega 10 modelos treinados (.h5), avalia cada um, seleciona o melhor,
converte para TFLite e realiza inferência comparativa entre o modelo
TensorFlow e o modelo TFLite.

Dataset: Fashion MNIST
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# ─── Configuração ─────────────────────────────────────────────────────────────
NUM_MODELOS = 10
MODELS_DIR = './'                           # Diretório com modelo1.h5 ... modelo10.h5
BEST_MODEL_FILE = 'modelo8.h5'              # Melhor modelo identificado
TFLITE_OUTPUT = 'modeloTeste.tflite'
RESULTS_FILE = 'arquivo_resultado.txt'
PREDICTIONS_FILE = 'predicoesTensorFlow.txt'
# ──────────────────────────────────────────────────────────────────────────────

# Carregamento do dataset
fashion_mnist = keras.datasets.fashion_mnist
(input_train, target_train), (input_test, target_test) = fashion_mnist.load_data()

class_names = ['Camiseta/Top', 'Calça', 'Sueter', 'Vestido', 'Casaco',
               'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota']

# Avaliação de todos os modelos salvos
modelos = [None] * NUM_MODELOS
for i in range(NUM_MODELOS):
    model_path = f'{MODELS_DIR}modelo{i+1}.h5'
    modelos[i] = tf.keras.models.load_model(model_path)
    loss, acc = modelos[i].evaluate(input_test, target_test, verbose=0)
    with open(RESULTS_FILE, "a") as arquivo:
        arquivo.write(f"\n Modelo{i+1} = Loss: {loss}  Acc: {acc} \n")

# Carregando o melhor modelo para predições detalhadas
model = tf.keras.models.load_model(f'{MODELS_DIR}{BEST_MODEL_FILE}')
predictions = model.predict(input_test)

# Predições por classe com o modelo TensorFlow
for i in range(10):
    erros = 0
    acertos = 0
    for x in range(len(input_test)):
        if target_test[x] == i:
            if np.argmax(predictions[x]) == target_test[x]:
                acertos += 1
            else:
                erros += 1
    with open(PREDICTIONS_FILE, "a") as pred:
        total = acertos + erros
        pct = (acertos / total) * 100 if total > 0 else 0
        pred.write(f'Classe {i}: acertos={acertos}, erros={erros} — '
                   f'Porcentagem={pct:.2f}%\n')

# Conversão para TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflmodel = converter.convert()
with open(TFLITE_OUTPUT, 'wb') as f:
    f.write(tflmodel)

# Inferência com modelo TFLite
interpreter = tf.lite.Interpreter(TFLITE_OUTPUT)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

floating_model = input_details[0]['dtype'] == np.float32
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

# Inferência em uma imagem de teste
img = input_test[0]
input_data = np.expand_dims(img, axis=(0, 3))
if floating_model:
    input_data = np.float32(input_data) / np.max(input_data)

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
results = np.squeeze(output_data)
top_k = results.argsort()[-5:][::-1]

for i in top_k:
    if floating_model:
        print(f'{float(results[i]):08.6f}: {class_names[i]}')
    else:
        print(f'{float(results[i] / 255.0):08.6f}: {class_names[i]}')
