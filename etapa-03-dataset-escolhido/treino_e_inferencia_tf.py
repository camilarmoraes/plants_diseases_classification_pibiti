"""
treino_e_inferencia_tf.py — Treino CNN personalizada e inferência comparativa.

Pipeline completo no dataset Plant Pathology 2020:
    1. Treino de CNN com 4 camadas Conv2D (100 epochs)
    2. Inferência com modelo TensorFlow
    3. Inferência com modelo TFLite
    4. Inferência com modelo TFLite Otimizado (quantizado)
    5. Exportação de resultados em CSV

Dataset: Plant Pathology 2020 (4 classes)
"""

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from keras_preprocessing import image

# ─── Configuração ─────────────────────────────────────────────────────────────
TRAIN_CSV = './train.csv'
TEST_CSV = './test.csv'
IMAGES_DIR = './images'                        # Diretório com todas as imagens
DATASET_DIR = './temp/images'                  # Diretório organizado por classes
MODEL_OUTPUT = './modelos/tensorflowCNN.h5'
TFLITE_OUTPUT = './modelos/tensorflowliteCNN.tflite'
TFLITE_QUANT_OUTPUT = './modelos/tensorflowliteOtimizadoCNN.tflite'
CSV_TF = './InferenciaTFCNN.csv'
CSV_TFLITE = './InferenciaTFLITECNN.csv'
CSV_TFLITE_QUANT = './InferenciaTFLITETIMIZADOCNN.csv'

BATCH_SIZE = 32
IMG_HEIGHT = 180
IMG_WIDTH = 180
EPOCHS = 100
VALIDATION_SPLIT = 0.2
# ──────────────────────────────────────────────────────────────────────────────

# Carregamento dos índices de treino
df = pd.read_csv(TRAIN_CSV, index_col=0)

# Carregamento das imagens organizadas por classes
train_images = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=VALIDATION_SPLIT, subset="training",
    seed=123, image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
    label_mode='categorical')

val_images = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=VALIDATION_SPLIT, subset="validation",
    seed=123, image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
    label_mode='categorical')

class_names = train_images.class_names
num_classes = len(class_names)

# Arquitetura da CNN (4 camadas convolucionais)
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(train_images, validation_data=val_images, epochs=EPOCHS)
model.save(MODEL_OUTPUT)


# ─── Preparação para inferência ──────────────────────────────────────────────
test_set = pd.read_csv(TEST_CSV, index_col=0)

X_test = []
for index, data in test_set.iterrows():
    filepath = os.path.join(IMAGES_DIR, index + ".jpg")
    img = image.load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    X_test.append(x)
X_test = np.vstack(X_test)


# ─── Inferência TensorFlow ──────────────────────────────────────────────────
predictions = model.predict(X_test, batch_size=10)
score = np.array(tf.nn.softmax(predictions))
df_out = pd.concat(
    [test_set.reset_index(),
     pd.DataFrame(score, columns=class_names)],
    axis=1).set_index("image_id")
df_out.to_csv(CSV_TF)


# ─── Inferência TFLite ──────────────────────────────────────────────────────
def run_tflite_inference(model_path, X_test, class_names, output_csv):
    """Executa inferência TFLite e exporta resultados em CSV."""
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.resize_tensor_input(
        input_details[0]['index'], (len(X_test), IMG_HEIGHT, IMG_WIDTH, 3))
    interpreter.resize_tensor_input(
        output_details[0]['index'], (len(X_test), len(class_names)))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], X_test)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])
    df_out = pd.concat(
        [test_set.reset_index(),
         pd.DataFrame(preds, columns=class_names)],
        axis=1).set_index("image_id")
    df_out.to_csv(output_csv)
    return preds


run_tflite_inference(TFLITE_OUTPUT, X_test, class_names, CSV_TFLITE)
run_tflite_inference(TFLITE_QUANT_OUTPUT, X_test, class_names, CSV_TFLITE_QUANT)
