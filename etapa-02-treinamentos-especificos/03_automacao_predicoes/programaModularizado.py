"""
programaModularizado.py — Versão modularizada da avaliação de modelos.

Pipeline automatizado com funções separadas para:
    1. Avaliar múltiplos modelos e selecionar o melhor
    2. Realizar predições por classe
    3. Converter para TFLite
    4. Inferência com TFLite

Dataset: Fashion MNIST
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

# ─── Configuração ─────────────────────────────────────────────────────────────
NUM_MODELOS = 10
MODELS_DIR = './'                     # Diretório com modelo1.h5 ... modelo10.h5
TFLITE_OUTPUT = 'modelo_convertido.tflite'
EVALUATE_RESULTS = 'EvaluateModularizado.txt'
# ──────────────────────────────────────────────────────────────────────────────

# Carregamento do dataset
fashion_mnist = keras.datasets.fashion_mnist
(input_train, target_train), (input_test, target_test) = fashion_mnist.load_data()


def modelEvaluate():
    """Avalia todos os modelos salvos e retorna o de melhor acurácia."""
    modelos = []
    melhorAcc = 0.0
    melhorModelo = None

    for i in range(NUM_MODELOS):
        model_path = f'{MODELS_DIR}modelo{i+1}.h5'
        modelos.append(tf.keras.models.load_model(model_path))
        loss, acc = modelos[i].evaluate(input_test, target_test, verbose=0)

        if melhorAcc < acc:
            melhorAcc = acc
            melhorModelo = modelos[i]

        with open(EVALUATE_RESULTS, "a") as arquivo:
            arquivo.write(f"\n Modelo{i+1} = Loss: {loss}  Acc: {acc} \n")

    return melhorModelo


def modelPredict(model):
    """
    Realiza predições por classe e salva resultados.

    Se model é np.ndarray, trata como predições TFLite já computadas.
    Caso contrário, roda model.predict() no dataset de teste.
    """
    if isinstance(model, np.ndarray):
        predictions = model
        arquivo = 'predicoes_tflite'
    else:
        predictions = model.predict(input_test)
        arquivo = 'predicoes_tf'

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
        with open(f"{arquivo}.txt", "a") as pred:
            pred.write(f'Classe {i}: acertos={acertos}, erros={erros} '
                       f'— Porcentagem={pct:.2f}%\n')


def convertTFLITE(model):
    """Converte modelo Keras para TFLite e salva o binário."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflmodel = converter.convert()
    with open(TFLITE_OUTPUT, 'wb') as f:
        f.write(tflmodel)
    return TFLITE_OUTPUT


def inferenceTFLITE(tflite_path):
    """Realiza inferência em lote com o modelo TFLite."""
    interpreter = tf.lite.Interpreter(tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Redimensiona para processar todas as imagens de teste de uma vez
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

    modelPredict(model_predictions)


# ─── Execução do pipeline ────────────────────────────────────────────────────
if __name__ == '__main__':
    model = modelEvaluate()
    modelPredict(model)
    tflite_path = convertTFLITE(model)
    inferenceTFLITE(tflite_path)