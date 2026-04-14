"""
filtro_cores_amarelas.py — Filtragem de cores amarelas/alaranjadas.

Filtra pixels de cores amarelas e alaranjadas de uma imagem já
pré-processada (com filtro de histograma aplicado). Preserva apenas
pixels onde:
  - Canal vermelho (R): 130–255
  - Canal verde (G): 0–200

Usado para isolar regiões de ferrugem (rust) e doenças múltiplas
nas folhas de macieira.
"""

import cv2
import numpy as np

# ─── Configuração ─────────────────────────────────────────────────────────────
ORIGINAL_IMAGE = './train_images/Train_338.jpg'
FILTERED_IMAGE = './filtrada/img3.jpg'          # Imagem após filtro de histograma

# Faixas de pixel para filtragem de cores alaranjadas
R_MIN, R_MAX = 130, 255                        # Canal vermelho
G_MIN, G_MAX = 0, 200                          # Canal verde
# ──────────────────────────────────────────────────────────────────────────────

imgOriginal = cv2.imread(ORIGINAL_IMAGE)
img = cv2.imread(FILTERED_IMAGE)
new_img = np.zeros(img.shape, dtype=np.uint8)

# Filtragem pixel a pixel
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        r_val = img[i][j][2]  # Canal vermelho (índice 2 em BGR)
        g_val = img[i][j][1]  # Canal verde (índice 1 em BGR)

        if (R_MIN <= r_val <= R_MAX) and (G_MIN <= g_val <= G_MAX):
            new_img[i][j] = img[i][j]

cv2.imshow('Original', imgOriginal)
cv2.imshow('Filtro de Histograma', img)
cv2.imshow('Filtro de Cores', new_img)
cv2.waitKey(0)
cv2.destroyAllWindows()