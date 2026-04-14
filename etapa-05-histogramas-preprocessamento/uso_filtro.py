"""
uso_filtro.py — Filtro de corte por canal para uma imagem individual.

Aplica filtros de corte de intensidade por canal (R, G, B) em uma
imagem individual. Os parâmetros de corte variam por classe de doença
(ver comentários no código para valores de cada classe).

Útil para visualização interativa e ajuste de parâmetros antes de
aplicar em lote (veja filtros_concatenacao.py).
"""

import cv2
import numpy as np

# ─── Configuração ─────────────────────────────────────────────────────────────
INPUT_IMAGE = './train_images/Train_338.jpg'

# ─── Parâmetros de corte por classe (descomente a classe desejada) ───────────
# Scab (Sarna):
# pixelCorteInicio_azul = 130
# pixelCorteFim_azul = 130
# pixelCorteInicio_verde = 100
# pixelCorteFim_verde = 130
# pixelCorteInicio_vermelho = 80
# pixelCorteFim_vermelho = 130

# Rust (Ferrugem):
# pixelCorteInicio_azul = 100
# pixelCorteFim_azul = 100
# pixelCorteInicio_verde = 200
# pixelCorteFim_verde = 230
# pixelCorteInicio_vermelho = 180
# pixelCorteFim_vermelho = 240

# Multiple Diseases:
# pixelCorteInicio_azul = 100
# pixelCorteFim_azul = 100
# pixelCorteInicio_verde = 170
# pixelCorteFim_verde = 223
# pixelCorteInicio_vermelho = 170
# pixelCorteFim_vermelho = 240

# Healthy (Saudável):
pixelCorteInicio_azul = 100
pixelCorteFim_azul = 140
pixelCorteInicio_verde = 100
pixelCorteFim_verde = 140
pixelCorteInicio_vermelho = 100
pixelCorteFim_vermelho = 140
# ──────────────────────────────────────────────────────────────────────────────


def pixelVal2(pix, r1, s1, r2, s2):
    """Corte de histograma: preserva apenas a faixa [r1, r2], zera o resto."""
    if 0 <= pix <= r1:
        return 0
    elif r1 < pix <= r2:
        return ((s2 - s1) / (r2 - r1)) * (pix - r1) + s1
    else:
        return 0


imgOriginal = cv2.imread(INPUT_IMAGE)
(B, G, R) = cv2.split(imgOriginal)

pixelVal_vec2 = np.vectorize(pixelVal2)

B = np.uint8(pixelVal_vec2(B, pixelCorteInicio_azul, 255,
             pixelCorteFim_azul, 255))
G = np.uint8(pixelVal_vec2(G, pixelCorteInicio_verde, 255,
             pixelCorteFim_verde, 255))
R = np.uint8(pixelVal_vec2(R, pixelCorteInicio_vermelho, 255,
             pixelCorteFim_vermelho, 255))

img = cv2.merge([B, G, R])

cv2.imshow('Filtrada', img)
cv2.imshow('Original', imgOriginal)
cv2.waitKey(0)
cv2.destroyAllWindows()