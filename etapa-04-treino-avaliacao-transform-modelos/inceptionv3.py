"""
inceptionv3.py — Treino com Inception V3 (Transfer Learning).

Utiliza InceptionV3 pré-treinada (ImageNet) com fine-tuning para classificar
folhas de macieira em 4 classes. Inclui conversão para TFLite/TFLite
quantizado e cálculo de acurácia.

Dataset: Plant Pathology 2020 (4 classes)
Arquitetura: InceptionV3 (base) → Flatten → Dense(512) → Dense(4)
"""

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.applications import InceptionV3
from keras_preprocessing import image

# ─── Configuração ─────────────────────────────────────────────────────────────
TRAIN_CSV = './train.csv'
TEST_CSV = './test.csv'
IMAGES_DIR = './images'
DATASET_DIR = './temp/images'

MODEL_OUTPUT = './modelos/InceptionV3.h5'
TFLITE_OUTPUT = './modelos/TFLITEInceptionV3.tflite'
TFLITE_QUANT_OUTPUT = './modelos/TFLITEOtimizadoInceptionV3.tflite'
CSV_RESULTS = './inferenciaInceptionV3.csv'
ACC_TFLITE_FILE = './modelos/AcuraciaInceptionV3.txt'
ACC_TFLITE_QUANT_FILE = './modelos/AcuraciaOTIMInceptionV3.txt'

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224
NUM_CLASSES = 4
EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 35
LEARNING_RATE = 2e-5
# ──────────────────────────────────────────────────────────────────────────────

# Carregamento dos dados
df = pd.read_csv(TRAIN_CSV, index_col=0)

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="training", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
    label_mode='categorical')

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=0.2, subset="validation", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
    label_mode='categorical')

class_names = train_ds.class_names
num_classes = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical",
                      input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# ─── Construção do modelo (Transfer Learning) ───────────────────────────────
base_model = InceptionV3(
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
    include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=output)

# Fine-tuning
TRAINABLE_LAYERS = len(model.layers) - len(base_model.layers) + 5
for layer in model.layers[:-TRAINABLE_LAYERS]:
    layer.trainable = False
for layer in model.layers[-TRAINABLE_LAYERS:]:
    layer.trainable = True

early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=EARLY_STOPPING_PATIENCE, monitor="val_accuracy",
    verbose=2, mode="auto")
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_OUTPUT, monitor="val_accuracy", mode="auto",
    verbose=2, save_best_only=True)

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
    metrics=['accuracy'])

model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
          callbacks=[early_stopping, checkpoint])


# ─── Conversão para TFLite ───────────────────────────────────────────────────
MODEL = tf.keras.models.load_model(MODEL_OUTPUT)

converter = tf.lite.TFLiteConverter.from_keras_model(MODEL)
with open(TFLITE_OUTPUT, 'wb') as f:
    f.write(converter.convert())

converter_quant = tf.lite.TFLiteConverter.from_keras_model(MODEL)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
with open(TFLITE_QUANT_OUTPUT, 'wb') as f:
    f.write(converter_quant.convert())


# ─── Preparação dos dados de teste ───────────────────────────────────────────
test_set = pd.read_csv(TEST_CSV, index_col=0)

X_test = []
for index, data in test_set.iterrows():
    filepath = os.path.join(IMAGES_DIR, index + ".jpg")
    img = image.load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    X_test.append(x)
X_test = np.vstack(X_test)

predictions = model.predict(X_test)
tf_pred_df = pd.DataFrame(predictions, columns=class_names)


# ─── Inferência TFLite e cálculo de acurácia ─────────────────────────────────
def run_tflite_and_accuracy(model_path, X_test, test_set, class_names,
                            acc_file):
    """Executa inferência TFLite e calcula acurácia versus ground truth."""
    interpreter = tf.lite.Interpreter(model_path)
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
    preds = interpreter.get_tensor(output_details[0]['index'])

    acertos = sum(1 for i, j in enumerate(test_set["Valores por Coluna"])
                  if np.argmax(preds[i]) == j)
    acc = acertos / len(test_set)
    print(f'Acc = {acc:.4f}')
    with open(acc_file, "a") as f:
        f.write(f'Acc = {acc:.4f}\n')
    return preds


tflite_preds = run_tflite_and_accuracy(
    TFLITE_OUTPUT, X_test, test_set, class_names, ACC_TFLITE_FILE)
tflite_quant_preds = run_tflite_and_accuracy(
    TFLITE_QUANT_OUTPUT, X_test, test_set, class_names, ACC_TFLITE_QUANT_FILE)

all_models_df = pd.concat(
    [tf_pred_df,
     pd.DataFrame(tflite_preds, columns=class_names),
     pd.DataFrame(tflite_quant_preds, columns=class_names)],
    keys=['TF Model', 'TFLite', 'TFLite quantized'], axis='columns')
all_models_df.to_csv(CSV_RESULTS)