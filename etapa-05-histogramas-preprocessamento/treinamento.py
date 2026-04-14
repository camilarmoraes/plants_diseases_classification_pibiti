"""
treinamento.py — Treino de CNN com imagens concatenadas.

Treina uma CNN com data augmentation usando imagens concatenadas
(4 filtros lado a lado: 400×1600). Inclui conversão para TFLite
e inferência comparativa.

Dataset: Imagens concatenadas do Plant Pathology 2020 (4 classes)
"""

import numpy as np
import pandas as pd
import os
import shutil
from shutil import copyfile
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from keras_preprocessing import image

# ─── Configuração ─────────────────────────────────────────────────────────────
TRAIN_CSV = './train.csv'
TEST_CSV = './test.csv'
CONCAT_IMAGES_DIR = './Imagens_Concatenadas'   # Diretório com imagens concatenadas
TEMP_DIR = './temporaria/images'
MODEL_OUTPUT = './Nova.h5'
TFLITE_OUTPUT = './Concatenada.tflite'
CSV_TF = './ResultadoNovo.csv'
CSV_TFLITE = './ResultadoConcatenadasTFLITE.csv'

BATCH_SIZE = 32
IMG_HEIGHT = 400
IMG_WIDTH = 1600
NUM_CLASSES = 4
EPOCHS = 1000
EARLY_STOPPING_PATIENCE = 35
# ──────────────────────────────────────────────────────────────────────────────

# Preparação do diretório temporário organizado por classes
df = pd.read_csv(TRAIN_CSV, index_col=0)

if os.path.exists('temporaria'):
    shutil.rmtree('temporaria')

for class_name in ['healthy', 'multiple_diseases', 'rust', 'scab']:
    os.makedirs(os.path.join(TEMP_DIR, class_name), exist_ok=True)

for index, data in df.iterrows():
    label = df.columns[np.argmax(data)]
    filepath = os.path.join(CONCAT_IMAGES_DIR, index + ".jpg")
    destination = os.path.join(TEMP_DIR, label, index + ".jpg")
    if os.path.exists(filepath):
        copyfile(filepath, destination)

for subdir in os.listdir(TEMP_DIR):
    print(subdir, len(os.listdir(os.path.join(TEMP_DIR, subdir))))

# Carregamento dos dados
train_ds = tf.keras.utils.image_dataset_from_directory(
    TEMP_DIR, validation_split=0.2, subset="training", seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE,
    label_mode='categorical')

val_ds = tf.keras.utils.image_dataset_from_directory(
    TEMP_DIR, validation_split=0.2, subset="validation", seed=123,
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


# ─── Plot de resultados ─────────────────────────────────────────────────────
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# ─── Inferência e acurácia ──────────────────────────────────────────────────
test_set = pd.read_csv(TEST_CSV, index_col=0)

X_test = []
for index, data in test_set.iterrows():
    filepath = os.path.join(CONCAT_IMAGES_DIR, index + ".jpg")
    img = image.load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    X_test.append(x)
X_test = np.vstack(X_test)

predictions = model.predict(X_test)
tf_pred_df = pd.DataFrame(predictions, columns=class_names)
tf_pred_df.to_csv(CSV_TF)

# Cálculo de acurácia TF
acertos = sum(1 for i, j in enumerate(test_set["Valores por Coluna"])
              if np.argmax(predictions[i]) == j)
total = len(test_set)
acc = acertos / total
print(f'Acurácia TF: {acc:.4f}')


# ─── Inferência TFLite ──────────────────────────────────────────────────────
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflmodel = converter.convert()
with open(TFLITE_OUTPUT, 'wb') as f:
    f.write(tflmodel)

interpreter = tf.lite.Interpreter(TFLITE_OUTPUT)
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

tflite_preds = interpreter.get_tensor(output_details[0]['index'])
tflite_pred_df = pd.DataFrame(tflite_preds, columns=class_names)
tflite_pred_df.to_csv(CSV_TFLITE)
