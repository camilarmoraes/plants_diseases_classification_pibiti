"""
usando_filtros_duplos.py — Aplicação de filtros duplos em lote.

Aplica dois filtros de cor em sequência para processar todas as
imagens do dataset:

  1. Filtro Rust: Preserva pixels com R ≥ 130 e G ≤ 200
     (tons alaranjados característicos de ferrugem)

  2. Filtro Scab: Preserva pixels com G ≥ 120 e R ≥ 120
     (tons amarelo-esverdeados característicos de sarna)

Processa todas as imagens em lote e salva nas pastas correspondentes.
"""

import cv2
import numpy as np
import os

# ─── Configuração ─────────────────────────────────────────────────────────────
# Imagens pré-processadas com filtro de histograma 'Rust' e 'Scab'
RUST_INPUT_DIR = './Images/r/'
SCAB_INPUT_DIR = './Images/s/'

# Diretórios de saída
RUST_OUTPUT_DIR = './Images/Rust/'
SCAB_OUTPUT_DIR = './Images/Scab/'

IMAGE_SIZE = (400, 400)

# Faixas de filtragem
# Rust: preserva alaranjados (R alto, G moderado)
RUST_R_MIN, RUST_R_MAX = 130, 255
RUST_G_MIN, RUST_G_MAX = 0, 200

# Scab: preserva amarelo-esverdeados (G e R moderados/altos)
SCAB_G_MIN, SCAB_G_MAX = 120, 255
SCAB_R_MIN, SCAB_R_MAX = 120, 255
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(RUST_OUTPUT_DIR, exist_ok=True)
os.makedirs(SCAB_OUTPUT_DIR, exist_ok=True)


# ─── Filtro Rust ─────────────────────────────────────────────────────────────
for arquivo in os.listdir(RUST_INPUT_DIR):
    new_img = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
    img = cv2.imread(os.path.join(RUST_INPUT_DIR, arquivo))
    if img is None:
        continue

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            r_val = img[i][j][2]
            g_val = img[i][j][1]
            if (RUST_R_MIN <= r_val <= RUST_R_MAX) and \
               (RUST_G_MIN <= g_val <= RUST_G_MAX):
                new_img[i][j] = img[i][j]

    cv2.imwrite(os.path.join(RUST_OUTPUT_DIR, arquivo), new_img)


# ─── Filtro Scab ─────────────────────────────────────────────────────────────
for arquivo in os.listdir(SCAB_INPUT_DIR):
    new_img = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
    img = cv2.imread(os.path.join(SCAB_INPUT_DIR, arquivo))
    if img is None:
        continue

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            g_val = img[i][j][1]
            r_val = img[i][j][2]
            if (SCAB_G_MIN <= g_val <= SCAB_G_MAX) and \
               (SCAB_R_MIN <= r_val <= SCAB_R_MAX):
                new_img[i][j] = img[i][j]

    cv2.imwrite(os.path.join(SCAB_OUTPUT_DIR, arquivo), new_img)
