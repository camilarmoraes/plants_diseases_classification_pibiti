"""
pedra_papel_tesoura.py — Classificação de gestos (Pedra, Papel, Tesoura).

Treina uma CNN com 4 camadas convolucionais e data augmentation para
classificar imagens de gestos de mão em 3 classes.

Dataset: Rock Paper Scissors (TensorFlow Datasets)
"""

import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

# ─── Configuração ─────────────────────────────────────────────────────────────
RPS_TRAIN_ZIP = './rps.zip'
RPS_TEST_ZIP = './rps-test-set.zip'
EXTRACT_DIR = './rps_extracted'
IMAGE_SIZE = (150, 150)
BATCH_SIZE = 126
EPOCHS = 25
MODEL_OUTPUT = './rps.h5'
# ──────────────────────────────────────────────────────────────────────────────


# Extração dos dados
for zip_path in [RPS_TRAIN_ZIP, RPS_TEST_ZIP]:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

# Diretórios das classes
rock_dir = os.path.join(EXTRACT_DIR, 'rps', 'rock')
paper_dir = os.path.join(EXTRACT_DIR, 'rps', 'paper')
scissors_dir = os.path.join(EXTRACT_DIR, 'rps', 'scissors')

print('Total training rock images:', len(os.listdir(rock_dir)))
print('Total training paper images:', len(os.listdir(paper_dir)))
print('Total training scissors images:', len(os.listdir(scissors_dir)))

# Visualização de exemplos
rock_files = os.listdir(rock_dir)
paper_files = os.listdir(paper_dir)
scissors_files = os.listdir(scissors_dir)

pic_index = 2
sample_paths = (
    [os.path.join(rock_dir, f) for f in rock_files[:pic_index]] +
    [os.path.join(paper_dir, f) for f in paper_files[:pic_index]] +
    [os.path.join(scissors_dir, f) for f in scissors_files[:pic_index]]
)
for img_path in sample_paths:
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()

# Data augmentation para treinamento
training_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = training_datagen.flow_from_directory(
    os.path.join(EXTRACT_DIR, 'rps'),
    target_size=IMAGE_SIZE, class_mode='categorical', batch_size=BATCH_SIZE)

validation_generator = validation_datagen.flow_from_directory(
    os.path.join(EXTRACT_DIR, 'rps-test-set'),
    target_size=IMAGE_SIZE, class_mode='categorical', batch_size=BATCH_SIZE)

# Arquitetura da CNN (4 camadas convolucionais)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           input_shape=(*IMAGE_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(train_generator, epochs=EPOCHS, steps_per_epoch=20,
                    validation_data=validation_generator, verbose=1,
                    validation_steps=3)

model.save(MODEL_OUTPUT)

# Plot de acurácia
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

plt.plot(range(len(acc)), acc, 'r', label='Training accuracy')
plt.plot(range(len(val_acc)), val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()
plt.show()
