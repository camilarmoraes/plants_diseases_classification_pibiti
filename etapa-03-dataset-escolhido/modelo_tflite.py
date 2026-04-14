"""
modelo_tflite.py — Conversão de modelo Keras para TFLite.

Carrega um modelo treinado (.h5) e gera duas versões TFLite:
    1. TFLite padrão
    2. TFLite otimizado (quantizado com Optimize.DEFAULT)
"""

import tensorflow as tf

# ─── Configuração ─────────────────────────────────────────────────────────────
MODEL_INPUT = './modelos/tensorflowCNN.h5'
TFLITE_OUTPUT = './modelos/tensorflowliteCNN.tflite'
TFLITE_QUANT_OUTPUT = './modelos/tensorflowliteOtimizadoCNN.tflite'
# ──────────────────────────────────────────────────────────────────────────────

# Carrega o modelo TensorFlow
model = tf.keras.models.load_model(MODEL_INPUT)

# Conversão para TFLite padrão
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflmodel = converter.convert()
with open(TFLITE_OUTPUT, 'wb') as f:
    f.write(tflmodel)
print(f'Modelo TFLite salvo em: {TFLITE_OUTPUT}')

# Conversão para TFLite otimizado (quantizado)
converter_quant = tf.lite.TFLiteConverter.from_keras_model(model)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
tflmodel_quant = converter_quant.convert()
with open(TFLITE_QUANT_OUTPUT, 'wb') as f:
    f.write(tflmodel_quant)
print(f'Modelo TFLite Quantizado salvo em: {TFLITE_QUANT_OUTPUT}')