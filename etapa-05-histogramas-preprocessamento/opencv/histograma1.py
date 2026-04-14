"""
histograma1.py — Análise de histograma RGB e HSV de uma imagem.

Calcula e plota histogramas nos espaços de cor BGR e HSV para uma
imagem individual do dataset, permitindo visualizar a distribuição
de cada canal.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ─── Configuração ─────────────────────────────────────────────────────────────
INPUT_IMAGE = './Test_0.jpg'
# ──────────────────────────────────────────────────────────────────────────────

img = cv2.imread(INPUT_IMAGE)

# Histograma BGR
b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
plt.plot(hist_r, color='r', label="R")
plt.plot(hist_g, color='g', label="G")
plt.plot(hist_b, color='b', label="B")
plt.legend()
plt.title('Histograma BGR')
plt.show()

# Histograma HSV
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
h, s, v = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
hist_h = cv2.calcHist([h], [0], None, [256], [0, 256])
hist_s = cv2.calcHist([s], [0], None, [256], [0, 256])
hist_v = cv2.calcHist([v], [0], None, [256], [0, 256])
plt.plot(hist_h, color='r', label="H")
plt.plot(hist_s, color='g', label="S")
plt.plot(hist_v, color='b', label="V")
plt.legend()
plt.title('Histograma HSV')
plt.show()
