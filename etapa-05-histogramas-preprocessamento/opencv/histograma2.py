"""
histograma2.py — Equalização de histograma e Gaussian Blur com segmentação.

Duas demonstrações:
    1. Equalização de histograma com CDF normalizado
    2. Gaussian Blur + segmentação por cor HSV (inRange)

Permite visualizar como pré-processar imagens para isolar regiões
de interesse (ex: remover fundo escuro).
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

# ─── Configuração ─────────────────────────────────────────────────────────────
INPUT_IMAGE_GRAY = './Test_0.jpg'       # Imagem para equalização de histograma
INPUT_IMAGE_COLOR = './Train_110.jpg'   # Imagem para segmentação por cor

# Faixa HSV para segmentação de fundo escuro
HSV_LOWER = np.array([0, 0, 0], dtype="uint8")
HSV_UPPER = np.array([180, 255, 40], dtype="uint8")
# ──────────────────────────────────────────────────────────────────────────────


# ─── Parte 1: Equalização de histograma ──────────────────────────────────────
img = cv2.imread(INPUT_IMAGE_GRAY, 0)
hist, bins = np.histogram(img.flatten(), 256, [0, 256])
cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()

plt.plot(cdf_normalized, color='b')
plt.hist(img.flatten(), 256, [0, 256], color='r')
plt.xlim([0, 256])
plt.legend(('CDF', 'Histograma'), loc='upper left')
plt.title('Equalização de Histograma')
plt.show()


# ─── Parte 2: Gaussian Blur + Segmentação HSV ───────────────────────────────
image = cv2.imread(INPUT_IMAGE_COLOR)
blur = cv2.GaussianBlur(image, (5, 5), 0)
blur_hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

# Máscara: remove pixels escuros (fundo)
mask = cv2.inRange(blur_hsv, HSV_LOWER, HSV_UPPER)
mask = 255 - mask
output = cv2.bitwise_and(image, image, mask=mask)

cv2.imshow("Output", output)
cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()