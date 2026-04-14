"""
cnn.py — Treino de CNN personalizada e inferência TFLite.

Rede neural convolucional com 4 camadas Conv2D + Data Augmentation.
Inclui conversão para TFLite e TFLite quantizado, e cálculo de
acurácia para cada formato.

Dataset: Plant Pathology 2020 (4 classes)
Arquitetura: Conv2D(16) → Conv2D(32) → Conv2D(64) → Conv2D(128) → Dense(128)
"""

import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import layers
from keras_preprocessing import image

# ─── Configuração ─────────────────────────────────────────────────────────────
TRAIN_CSV = './train.csv'
TEST_CSV = './test.csv'
IMAGES_DIR = './images'
DATASET_DIR = './temp/images'                  # Organizado por classes

MODEL_OUTPUT = './modelos/CNN.h5'
TFLITE_OUTPUT = './modelos/TFLITECNN.tflite'
TFLITE_QUANT_OUTPUT = './modelos/TFLITEOtimizadoCNN.tflite'
CSV_RESULTS = './inferenciaCNN.csv'
ACC_TFLITE_FILE = './modelos/AcuraciaCNN.txt'
ACC_TFLITE_QUANT_FILE = './modelos/AcuraciaOTIMCNN.txt'

BATCH_SIZE = 32
IMG_HEIGHT = 200
IMG_WIDTH = 200
NUM_CLASSES = 4
EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 35
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
print(num_classes, class_names)

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

# ─── Arquitetura do modelo ───────────────────────────────────────────────────
model = tf.keras.Sequential([
    data_augmentation,
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=EARLY_STOPPING_PATIENCE, monitor="val_accuracy",
    verbose=2, mode="auto")
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_OUTPUT, monitor="val_accuracy", mode="auto",
    verbose=2, save_best_only=True)

model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])

history = model.fit(
    train_ds, validation_data=val_ds, epochs=EPOCHS,
    callbacks=[early_stopping, checkpoint])


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

# Predição TensorFlow
predictions = model.predict(X_test)
tf_pred_dataframe = pd.DataFrame(predictions)
tf_pred_dataframe.columns = class_names


# ─── Inferência TFLite ──────────────────────────────────────────────────────
def run_tflite_and_accuracy(model_path, X_test, test_set, class_names,
                            acc_file):
    """Executa inferência TFLite e calcula acurácia versus ground truth."""
    interpreter = tf.lite.Interpreter(model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.resize_tensor_input(
        input_details[0]['index'],
        (len(X_test), IMG_HEIGHT, IMG_WIDTH, 3))
    interpreter.resize_tensor_input(
        output_details[0]['index'], (len(X_test), NUM_CLASSES))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], X_test)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_details[0]['index'])

    # Cálculo de acurácia
    acertos = 0
    erros = 0
    for i, j in enumerate(test_set["Valores por Coluna"]):
        if np.argmax(preds[i]) == j:
            acertos += 1
        else:
            erros += 1

    total = erros + acertos
    acc = acertos / total
    print(f'Acc = {acc:.4f}')
    with open(acc_file, "a") as f:
        f.write(f'Acc = {acc:.4f}\n')

    return preds


# TFLite padrão
tflite_preds = run_tflite_and_accuracy(
    TFLITE_OUTPUT, X_test, test_set, class_names, ACC_TFLITE_FILE)
tflite_pred_df = pd.DataFrame(tflite_preds, columns=class_names)

# TFLite otimizado (quantizado)
tflite_quant_preds = run_tflite_and_accuracy(
    TFLITE_QUANT_OUTPUT, X_test, test_set, class_names, ACC_TFLITE_QUANT_FILE)
tflite_quant_pred_df = pd.DataFrame(tflite_quant_preds, columns=class_names)


# ─── Exportação de resultados comparativos ───────────────────────────────────
all_models_df = pd.concat(
    [tf_pred_dataframe, tflite_pred_df, tflite_quant_pred_df],
    keys=['TF Model', 'TFLite', 'TFLite quantized'], axis='columns')
all_models_df.to_csv(CSV_RESULTS)