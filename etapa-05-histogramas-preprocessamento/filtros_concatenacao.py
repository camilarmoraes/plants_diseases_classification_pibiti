"""
filtros_concatenacao.py — Aplicação de filtro de histograma em lote.

Aplica o filtro de corte por canal (Scab) em todas as imagens de um
diretório e salva os resultados filtrados.
"""

import cv2
import numpy as np
import os

# ─── Configuração ─────────────────────────────────────────────────────────────
INPUT_DIR = './400x400'                        # Imagens de entrada (400x400)
OUTPUT_DIR = './filtros_aplicados/'             # Imagens filtradas

# Parâmetros de corte (Scab)
PIXEL_CORTE_INICIO_AZUL = 130
PIXEL_CORTE_FIM_AZUL = 130
PIXEL_CORTE_INICIO_VERDE = 100
PIXEL_CORTE_FIM_VERDE = 130
PIXEL_CORTE_INICIO_VERMELHO = 80
PIXEL_CORTE_FIM_VERMELHO = 130
# ──────────────────────────────────────────────────────────────────────────────


def pixelVal2(pix, r1, s1, r2, s2):
    """Corte de histograma: preserva apenas a faixa [r1, r2], zera o resto."""
    if 0 <= pix <= r1:
        return 0
    elif r1 < pix <= r2:
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return 0


pixelVal_vec2 = np.vectorize(pixelVal2)
os.makedirs(OUTPUT_DIR, exist_ok=True)

for arquivo in os.listdir(INPUT_DIR):
    imgOriginal = cv2.imread(os.path.join(INPUT_DIR, arquivo))
    if imgOriginal is None:
        continue

    (B, G, R) = cv2.split(imgOriginal)

    B = np.uint8(pixelVal_vec2(B, PIXEL_CORTE_INICIO_AZUL, 255,
                 PIXEL_CORTE_FIM_AZUL, 255))
    G = np.uint8(pixelVal_vec2(G, PIXEL_CORTE_INICIO_VERDE, 255,
                 PIXEL_CORTE_FIM_VERDE, 255))
    R = np.uint8(pixelVal_vec2(R, PIXEL_CORTE_INICIO_VERMELHO, 255,
                 PIXEL_CORTE_FIM_VERMELHO, 255))

    img = cv2.merge([B, G, R])
    cv2.imwrite(os.path.join(OUTPUT_DIR, arquivo), img)
