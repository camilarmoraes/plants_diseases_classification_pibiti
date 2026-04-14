"""
modelo_simples.py — Treino de modelo simples (1 camada Conv2D).

Modelo baseline com uma única camada convolucional, sem data augmentation
e sem transfer learning. Serve como referência para comparação com
modelos mais complexos.

Dataset: Plant Pathology 2020 (4 classes)
Arquitetura: Conv2D(32) → Dense(128)
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
IMAGES_DIR = './images'
DATASET_DIR = './temp/images'

MODEL_OUTPUT = './modelos/TFSimples.h5'
TFLITE_OUTPUT = './modelos/TFLITESIMPLES.tflite'
TFLITE_QUANT_OUTPUT = './modelos/TFLITEOtimizadoSimples.tflite'
CSV_RESULTS = './inferenciaMODELOSIMPLES.csv'
ACC_TFLITE_FILE = './modelos/AcuraciaTFLITE.txt'
ACC_TFLITE_QUANT_FILE = './modelos/AcuraciaOTIM.txt'

IMG_HEIGHT = 200
IMG_WIDTH = 200
NUM_CLASSES = 4
EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 30
VALIDATION_SPLIT = 0.18
# ──────────────────────────────────────────────────────────────────────────────

# Carregamento dos dados
df = pd.read_csv(TRAIN_CSV, index_col=0)

imagens_treino = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=VALIDATION_SPLIT, subset="training",
    seed=123, image_size=(IMG_HEIGHT, IMG_WIDTH), label_mode='categorical')

imagens_validacao = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR, validation_split=VALIDATION_SPLIT, subset="validation",
    seed=123, image_size=(IMG_HEIGHT, IMG_WIDTH), label_mode="categorical")

class_names = imagens_treino.class_names
num_classes = len(class_names)
print(num_classes, class_names)

# Arquitetura simples (1 camada Conv2D)
model = keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu'),
    tf.keras.layers.MaxPooling2D(3, 3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=EARLY_STOPPING_PATIENCE, monitor="val_accuracy",
    verbose=2, mode="auto")
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_OUTPUT, monitor="val_accuracy", mode="auto",
    verbose=2, save_best_only=True)

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(imagens_treino, validation_data=imagens_validacao,
          epochs=EPOCHS, callbacks=[early_stopping, checkpoint])


# ─── Conversão para TFLite ───────────────────────────────────────────────────
MODEL = tf.keras.models.load_model(MODEL_OUTPUT)

converter = tf.lite.TFLiteConverter.from_keras_model(MODEL)
tflmodel = converter.convert()
with open(TFLITE_OUTPUT, 'wb') as f:
    f.write(tflmodel)

converter_quant = tf.lite.TFLiteConverter.from_keras_model(MODEL)
converter_quant.optimizations = [tf.lite.Optimize.DEFAULT]
tflmodel_quant = converter_quant.convert()
with open(TFLITE_QUANT_OUTPUT, 'wb') as f:
    f.write(tflmodel_quant)


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

# Predições TensorFlow
predictions = model.predict(X_test)
tf_pred_dataframe = pd.DataFrame(predictions, columns=class_names)


# ─── Inferência TFLite e cálculo de acurácia ─────────────────────────────────
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

    acertos = sum(1 for i, j in enumerate(test_set["Valores por Coluna"])
                  if np.argmax(preds[i]) == j)
    total = len(test_set)
    acc = acertos / total
    print(f'Acc = {acc:.4f}')
    with open(acc_file, "a") as f:
        f.write(f'Acc = {acc:.4f}\n')

    return preds


tflite_preds = run_tflite_and_accuracy(
    TFLITE_OUTPUT, X_test, test_set, class_names, ACC_TFLITE_FILE)
tflite_pred_df = pd.DataFrame(tflite_preds, columns=class_names)

tflite_quant_preds = run_tflite_and_accuracy(
    TFLITE_QUANT_OUTPUT, X_test, test_set, class_names, ACC_TFLITE_QUANT_FILE)
tflite_quant_pred_df = pd.DataFrame(tflite_quant_preds, columns=class_names)

# Exportação de resultados comparativos
all_models_df = pd.concat(
    [tf_pred_dataframe, tflite_pred_df, tflite_quant_pred_df],
    keys=['TF Model', 'TFLite', 'TFLite quantized'], axis='columns')
all_models_df.to_csv(CSV_RESULTS)