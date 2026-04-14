"""
filtros_morfologicos.py — Aplicação de filtros morfológicos e segmentação HSV.

Aplica filtros morfológicos (erosão, dilatação, opening, closing, gradient,
top hat, black hat) e segmentação por cor HSV (inRange) em lote em todas
as imagens do dataset organizadas por classe.

Os resultados são salvos em subpastas separadas por tipo de filtro.
"""

import cv2
import numpy as np
import os

# ─── Configuração ─────────────────────────────────────────────────────────────
# Diretórios de imagens por classe
INPUT_DIRS = {
    'healthy': './images/healthy',
    'multiple_diseases': './images/multiple_diseases',
    'rust': './images/rust',
    'scab': './images/scab',
}

# Diretório de saída
OUTPUT_DIR = './filtros/'

# Faixa HSV para segmentação de cores amarelas/alaranjadas (sintomas de doença)
HSV_LOWER = (15, 0, 0)
HSV_UPPER = (36, 255, 255)
# ──────────────────────────────────────────────────────────────────────────────


def apply_inrange_filter(img):
    """Aplica filtro de segmentação HSV + grayscale para preservar cores alvo."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    result = cv2.bitwise_and(img, img, mask=mask)

    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw_bgr = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    result2 = cv2.bitwise_or(bw_bgr, result)
    return result2


# Processamento por classe
output_inrange = os.path.join(OUTPUT_DIR, 'inRange')
os.makedirs(output_inrange, exist_ok=True)

for class_name, input_dir in INPUT_DIRS.items():
    if not os.path.exists(input_dir):
        print(f'Aviso: Diretório não encontrado: {input_dir}')
        continue

    for arquivo in os.listdir(input_dir):
        img = cv2.imread(os.path.join(input_dir, arquivo))
        if img is None:
            continue

        result = apply_inrange_filter(img)
        cv2.imwrite(os.path.join(output_inrange, arquivo), result)

    print(f'Classe {class_name} processada.')