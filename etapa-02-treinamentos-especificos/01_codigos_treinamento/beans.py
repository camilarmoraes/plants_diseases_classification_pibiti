"""
beans.py — Classificação de doenças em folhas de feijão com CNN.

Treina uma rede neural convolucional para classificar imagens de folhas
de feijão em 3 classes: angular_leaf_spot, bean_rust e healthy.

Dataset: https://github.com/AI-Lab-Makerere/ibean
"""

import os
import zipfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras

# ─── Configuração ─────────────────────────────────────────────────────────────
# Caminhos para os arquivos ZIP do dataset Beans
BEANS_TEST_ZIP = './beans/test.zip'
BEANS_TRAIN_ZIP = './beans/train.zip'
BEANS_VALIDATION_ZIP = './beans/validation.zip'

# Diretório para extração dos dados
EXTRACT_DIR = './beans_extracted'

IMAGE_SIZE = (500, 500)
BATCH_SIZE = 64
EPOCHS = 25
# ──────────────────────────────────────────────────────────────────────────────


# Extração dos dados
for zip_path in [BEANS_TEST_ZIP, BEANS_TRAIN_ZIP, BEANS_VALIDATION_ZIP]:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(EXTRACT_DIR)

# Contagem de imagens por classe
test_dir = os.path.join(EXTRACT_DIR, 'test')
angular_leaf_spot = os.path.join(test_dir, 'angular_leaf_spot')
bean_rust = os.path.join(test_dir, 'bean_rust')
healthy = os.path.join(test_dir, 'healthy')

print('Total training angular leaf spot images:', len(os.listdir(angular_leaf_spot)))
print('Total training rust images:', len(os.listdir(bean_rust)))
print('Total training healthy images:', len(os.listdir(healthy)))

# Visualização de exemplos
spot_files = os.listdir(angular_leaf_spot)
rust_files = os.listdir(bean_rust)
healthy_files = os.listdir(healthy)

pic_index = 2
sample_paths = (
    [os.path.join(angular_leaf_spot, f) for f in spot_files[:pic_index]] +
    [os.path.join(bean_rust, f) for f in rust_files[:pic_index]] +
    [os.path.join(healthy, f) for f in healthy_files[:pic_index]]
)
for img_path in sample_paths:
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.axis('Off')
    plt.show()

# Carregamento dos datasets
train_images = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(EXTRACT_DIR, 'train'),
    image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

test_images = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

validation_images = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE)

# Arquitetura da CNN
model = keras.Sequential([
    tf.keras.layers.Conv2D(32, (2, 2), activation='relu',
                           input_shape=(*IMAGE_SIZE, 3)),
    tf.keras.layers.MaxPooling2D(3, 3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='softmax')
])

model.summary()
model.compile(loss='categorical_crossentropy', optimizer='rmsprop',
              metrics=['accuracy'])

history = model.fit(train_images, epochs=EPOCHS,
                    validation_data=validation_images, verbose=1,
                    validation_steps=3)
