"""
densenet121_treino_inferencia.py — Treino com DenseNet121 (Transfer Learning).

Utiliza DenseNet121 pré-treinada (ImageNet) com fine-tuning para classificar
folhas de macieira em 4 classes de doenças. Inclui inferência com modelo
TFLite otimizado.

Dataset: Plant Pathology 2020 (4 classes)
"""

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import DenseNet121

# ─── Configuração ─────────────────────────────────────────────────────────────
TRAIN_CSV = './train.csv'
TEST_CSV = './test.csv'
IMAGES_DIR = './images'
DATASET_DIR = './temp/images'                   # Organizado por classes
MODEL_OUTPUT = './modelos/DenseNet121.h5'
TFLITE_QUANT_OUTPUT = './modelos/DenseNet121Otimizado.tflite'
CSV_TF = './DENSENETINFERENCIA.csv'
CSV_TFLITE_QUANT = './InferenciaTFLITEDENSENET121OTIMIZADO.csv'

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 4
EPOCHS = 40
LEARNING_RATE = 2e-5
# ──────────────────────────────────────────────────────────────────────────────

# Carregamento dos dados
df = pd.read_csv(TRAIN_CSV, index_col=0)

train_images = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="training", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
    label_mode='categorical')

val_images = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="validation", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
    label_mode='categorical')

class_names = train_images.class_names

# ─── Construção do modelo (Transfer Learning) ───────────────────────────────
base_model = DenseNet121(
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
    include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Fine-tuning: descongelar últimas camadas
TRAINABLE_LAYERS = len(model.layers) - len(base_model.layers) + 5
for layer in model.layers[:-TRAINABLE_LAYERS]:
    layer.trainable = False
for layer in model.layers[-TRAINABLE_LAYERS:]:
    layer.trainable = True

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
    metrics=['acc'])

model.fit(train_images, validation_data=val_images, epochs=EPOCHS)
model.save(MODEL_OUTPUT)


# ─── Inferência TensorFlow ──────────────────────────────────────────────────
test_set = pd.read_csv(TEST_CSV, index_col=0)

X_test = []
for index, data in test_set.iterrows():
    filepath = os.path.join(IMAGES_DIR, index + ".jpg")
    img = image.load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    X_test.append(x)
X_test = np.vstack(X_test)

predictions = model.predict(X_test, batch_size=10)
score = np.array(tf.nn.softmax(predictions))
df_out = pd.concat(
    [test_set.reset_index(),
     pd.DataFrame(score, columns=class_names)],
    axis=1).set_index("image_id")
df_out.to_csv(CSV_TF)


# ─── Inferência TFLite Otimizado ─────────────────────────────────────────────
interpreter = tf.lite.Interpreter(TFLITE_QUANT_OUTPUT)
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
df_out = pd.concat(
    [test_set.reset_index(),
     pd.DataFrame(model_predictions, columns=class_names)],
    axis=1).set_index("image_id")
df_out.to_csv(CSV_TFLITE_QUANT)
