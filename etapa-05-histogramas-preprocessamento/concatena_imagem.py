"""
concatena_imagem.py — Concatenação horizontal de imagens filtradas.

Concatena horizontalmente as imagens filtradas de cada classe:
  1. Healthy + Multiple Diseases → concatena1
  2. Rust + Scab → concatena2
  3. concatena1 + concatena2 → Imagem final (4 filtros lado a lado)

As imagens resultantes são usadas para treinar o modelo com múltiplas
representações visuais.
"""

import cv2
import os

# ─── Configuração ─────────────────────────────────────────────────────────────
INPUT_BASE = './Images/'

HEALTHY_DIR = os.path.join(INPUT_BASE, 'Healthy')
MULTIPLE_DIR = os.path.join(INPUT_BASE, 'Multiple_Diseases')
RUST_DIR = os.path.join(INPUT_BASE, 'Rust')
SCAB_DIR = os.path.join(INPUT_BASE, 'Scab')

CONCAT1_DIR = os.path.join(INPUT_BASE, 'concatena1')
CONCAT2_DIR = os.path.join(INPUT_BASE, 'concatena2')
FINAL_DIR = os.path.join(INPUT_BASE, 'Todas')
# ──────────────────────────────────────────────────────────────────────────────

# Criação das pastas de saída
for d in [CONCAT1_DIR, CONCAT2_DIR, FINAL_DIR]:
    os.makedirs(d, exist_ok=True)

# Passo 1: Healthy + Multiple Diseases
dir_healthy = set(os.listdir(HEALTHY_DIR))
dir_multiple = set(os.listdir(MULTIPLE_DIR))

for file in dir_healthy & dir_multiple:
    img1 = cv2.imread(os.path.join(HEALTHY_DIR, file))
    img2 = cv2.imread(os.path.join(MULTIPLE_DIR, file))
    if img1 is not None and img2 is not None:
        conc = cv2.hconcat([img1, img2])
        cv2.imwrite(os.path.join(CONCAT1_DIR, file), conc)

# Passo 2: Rust + Scab
dir_rust = set(os.listdir(RUST_DIR))
dir_scab = set(os.listdir(SCAB_DIR))

for file in dir_rust & dir_scab:
    img1 = cv2.imread(os.path.join(RUST_DIR, file))
    img2 = cv2.imread(os.path.join(SCAB_DIR, file))
    if img1 is not None and img2 is not None:
        conc = cv2.hconcat([img1, img2])
        cv2.imwrite(os.path.join(CONCAT2_DIR, file), conc)

# Passo 3: Concatenação final (4 filtros)
dir_con1 = set(os.listdir(CONCAT1_DIR))
dir_con2 = set(os.listdir(CONCAT2_DIR))

for file in dir_con1 & dir_con2:
    img1 = cv2.imread(os.path.join(CONCAT1_DIR, file))
    img2 = cv2.imread(os.path.join(CONCAT2_DIR, file))
    if img1 is not None and img2 is not None:
        conc = cv2.hconcat([img1, img2])
        cv2.imwrite(os.path.join(FINAL_DIR, file), conc)

print(f'Imagens concatenadas salvas em: {FINAL_DIR}')