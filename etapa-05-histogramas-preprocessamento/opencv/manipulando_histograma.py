"""
manipulando_histograma.py — Corte de histograma por faixas de pixel.

Aplica uma função de corte (pixelVal2) para isolar faixas específicas
de intensidade em cada canal (B, G, R). Divide a imagem em quadrantes
e calcula a intensidade média de cada um.

Útil para ajustar os parâmetros de filtro antes de aplicar em lote.
"""

import cv2
import numpy as np

# ─── Configuração ─────────────────────────────────────────────────────────────
INPUT_IMAGE = './Train_170.jpg'

# Faixas de corte por canal (ajustar conforme a doença alvo)
PIXEL_CORTE_INICIO_AZUL = 63
PIXEL_CORTE_FIM_AZUL = 173
PIXEL_CORTE_INICIO_VERDE = 95
PIXEL_CORTE_FIM_VERDE = 223
PIXEL_CORTE_INICIO_VERMELHO = 95
PIXEL_CORTE_FIM_VERMELHO = 250
# ──────────────────────────────────────────────────────────────────────────────


def pixelVal(pix, r1, s1, r2, s2):
    """Transformação linear por partes para ajuste de histograma."""
    if 0 <= pix <= r1:
        return (s1 / r1) * pix
    elif r1 < pix <= r2:
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return ((255 - s2) / (255 - r2)) * (pix - r2) + s2


def pixelVal2(pix, r1, s1, r2, s2):
    """Corte de histograma: preserva apenas a faixa [r1, r2], zera o resto."""
    if 0 <= pix <= r1:
        return 0
    elif r1 < pix <= r2:
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return 0


# Carregamento e processamento
imgOriginal = cv2.imread(INPUT_IMAGE)
(B, G, R) = cv2.split(imgOriginal)

pixelVal_vec2 = np.vectorize(pixelVal2)

B = np.uint8(pixelVal_vec2(B, PIXEL_CORTE_INICIO_AZUL,
             PIXEL_CORTE_INICIO_AZUL, PIXEL_CORTE_FIM_AZUL,
             PIXEL_CORTE_FIM_AZUL))
G = np.uint8(pixelVal_vec2(G, PIXEL_CORTE_INICIO_VERDE,
             PIXEL_CORTE_INICIO_VERDE, PIXEL_CORTE_FIM_VERDE,
             PIXEL_CORTE_FIM_VERDE))
R = np.uint8(pixelVal_vec2(R, PIXEL_CORTE_INICIO_VERMELHO,
             PIXEL_CORTE_INICIO_VERMELHO, PIXEL_CORTE_FIM_VERMELHO,
             PIXEL_CORTE_FIM_VERMELHO))

img = cv2.merge([B, G, R])

# Divisão em quadrantes e cálculo de intensidade média
height, width = img.shape[:2]
metadeAltura = height // 2
metadeLargura = width // 2

quadrantes = {
    'Superior Esquerdo': img[0:metadeAltura, 0:metadeLargura],
    'Inferior Esquerdo': img[metadeAltura:, 0:metadeLargura],
    'Superior Direito':  img[0:metadeAltura, metadeLargura:],
    'Inferior Direito':  img[metadeAltura:, metadeLargura:],
}

for nome, quad in quadrantes.items():
    media = np.round(np.mean(quad))
    print(f'Intensidade média {nome}: {media}')

cv2.imshow('Filtrada', img)
cv2.imshow('Original', imgOriginal)
cv2.waitKey(0)
cv2.destroyAllWindows()