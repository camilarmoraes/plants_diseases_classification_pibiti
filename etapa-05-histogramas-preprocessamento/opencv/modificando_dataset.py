"""
modificando_dataset.py — Geração de versões do dataset com canais removidos.

Cria 7 versões de cada imagem do dataset, removendo diferentes combinações
de canais de cor, para estudar a importância de cada canal na classificação.

Versões geradas:
    - BGR (original convertido)
    - No_Blue (sem canal azul)
    - No_Green (sem canal verde)
    - No_Red (sem canal vermelho)
    - No_Red_No_Green (apenas azul)
    - No_Red_No_Blue (apenas verde)
    - No_Green_No_Blue (apenas vermelho)
"""

import cv2 as cv
import os
import copy

# ─── Configuração ─────────────────────────────────────────────────────────────
INPUT_DIR = './images'                  # Diretório com imagens originais
OUTPUT_BASE = './canais_separados/'     # Diretório de saída

# Subpastas de saída (serão criadas automaticamente)
OUTPUT_DIRS = {
    'BGR': os.path.join(OUTPUT_BASE, 'BGR'),
    'No_Blue': os.path.join(OUTPUT_BASE, 'No_Blue'),
    'No_Green': os.path.join(OUTPUT_BASE, 'No_Green'),
    'No_Red': os.path.join(OUTPUT_BASE, 'No_Red'),
    'No_Red_No_Green': os.path.join(OUTPUT_BASE, 'No_Red_No_Green'),
    'No_Red_No_Blue': os.path.join(OUTPUT_BASE, 'No_Red_No_Blue'),
    'No_Green_No_Blue': os.path.join(OUTPUT_BASE, 'No_Green_No_Blue'),
}
# ──────────────────────────────────────────────────────────────────────────────

# Criação das pastas de saída
for dir_path in OUTPUT_DIRS.values():
    os.makedirs(dir_path, exist_ok=True)

for file in os.listdir(INPUT_DIR):
    x = cv.imread(os.path.join(INPUT_DIR, file))
    if x is None:
        continue
    img = cv.cvtColor(x, cv.COLOR_BGR2RGB)

    # Remove canal azul (índice 0 em RGB)
    img_no_blue = copy.copy(img)
    img_no_blue[:, :, 0] = 0

    # Remove canal verde (índice 1 em RGB)
    img_no_green = copy.copy(img)
    img_no_green[:, :, 1] = 0

    # Remove canal vermelho (índice 2 em RGB)
    img_no_red = copy.copy(img)
    img_no_red[:, :, 2] = 0

    # Combinações de remoção
    img_no_red_no_green = copy.copy(img)
    img_no_red_no_green[:, :, 1] = 0
    img_no_red_no_green[:, :, 2] = 0

    img_no_red_no_blue = copy.copy(img)
    img_no_red_no_blue[:, :, 2] = 0
    img_no_red_no_blue[:, :, 0] = 0

    img_no_green_no_blue = copy.copy(img)
    img_no_green_no_blue[:, :, 1] = 0
    img_no_green_no_blue[:, :, 0] = 0

    # Salvamento
    cv.imwrite(os.path.join(OUTPUT_DIRS['BGR'], file), img)
    cv.imwrite(os.path.join(OUTPUT_DIRS['No_Blue'], file), img_no_blue)
    cv.imwrite(os.path.join(OUTPUT_DIRS['No_Green'], file), img_no_green)
    cv.imwrite(os.path.join(OUTPUT_DIRS['No_Red'], file), img_no_red)
    cv.imwrite(os.path.join(OUTPUT_DIRS['No_Red_No_Green'], file), img_no_red_no_green)
    cv.imwrite(os.path.join(OUTPUT_DIRS['No_Red_No_Blue'], file), img_no_red_no_blue)
    cv.imwrite(os.path.join(OUTPUT_DIRS['No_Green_No_Blue'], file), img_no_green_no_blue)
