"""
inferencia_tflite.py — Inferência isolada com modelo TFLite.

Carrega imagens de teste e realiza inferência com um modelo TFLite
já convertido. Exporta os resultados em CSV.

Dataset: Plant Pathology 2020 (4 classes)
"""

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras_preprocessing import image

# ─── Configuração ─────────────────────────────────────────────────────────────
IMAGES_DIR = './images'
TEST_CSV = './test.csv'
TFLITE_MODEL = './modelos/tensorflowlite.tflite'
OUTPUT_CSV = './InferenciaTFLITE.csv'

IMG_HEIGHT = 180
IMG_WIDTH = 180
NUM_CLASSES = 4
CLASS_NAMES = ['healthy', 'multiple_diseases', 'rust', 'scab']
# ──────────────────────────────────────────────────────────────────────────────

# Preparação dos dados de teste
test_set = pd.read_csv(TEST_CSV, index_col=0)

X_test = []
for index, data in test_set.iterrows():
    filepath = os.path.join(IMAGES_DIR, index + ".jpg")
    img = image.load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    X_test.append(x)
X_test = np.vstack(X_test)
print(f'Imagens de teste carregadas: {len(X_test)}')

# Inferência TFLite
interpreter = tf.lite.Interpreter(TFLITE_MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.resize_tensor_input(
    input_details[0]['index'], (len(X_test), IMG_HEIGHT, IMG_WIDTH, 3))
interpreter.resize_tensor_input(
    output_details[0]['index'], (len(X_test), NUM_CLASSES))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], X_test)
interpreter.invoke()

model_predictions = interpreter.get_tensor(output_details[0]['index'])

# Exportação dos resultados
df_out = pd.concat(
    [test_set.reset_index(),
     pd.DataFrame(model_predictions, columns=CLASS_NAMES)],
    axis=1).set_index("image_id")
df_out.to_csv(OUTPUT_CSV)
print(f'Resultados salvos em: {OUTPUT_CSV}')
df_out.head()
