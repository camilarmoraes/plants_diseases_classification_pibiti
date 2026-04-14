"""
media_histograma.py — Cálculo do histograma médio por classe.

Lê os CSVs gerados pelo programa_histogramas.py e calcula o
histograma médio (R, G, B) para cada classe, permitindo identificar
padrões de cor característicos de cada doença.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# ─── Configuração ─────────────────────────────────────────────────────────────
# Classe a processar (alterar conforme necessário)
CLASS_NAME = 'Scab'
HISTOGRAMS_DIR = f'./Histogramas/{CLASS_NAME}/Tabelas/'
OUTPUT_CSV = f'./Histogramas/{CLASS_NAME}/resultado_medio.csv'
NUM_IMAGES = 622                               # Quantidade de imagens na classe
# ──────────────────────────────────────────────────────────────────────────────

cor_vermelha = np.zeros(256).tolist()
cor_azul = np.zeros(256).tolist()
cor_verde = np.zeros(256).tolist()

for filename in os.listdir(HISTOGRAMS_DIR):
    df = pd.read_csv(os.path.join(HISTOGRAMS_DIR, filename))

    for canal, lista in [('R', cor_vermelha), ('G', cor_verde), ('B', cor_azul)]:
        for i, val in enumerate(df[canal]):
            lista[i] += val

# Calcula a média
for lista in [cor_vermelha, cor_verde, cor_azul]:
    for i in range(len(lista)):
        lista[i] = lista[i] / NUM_IMAGES

# Salva resultado
df_result = pd.DataFrame({'R': cor_vermelha, 'G': cor_verde, 'B': cor_azul})
df_result.to_csv(OUTPUT_CSV)
print(f'Histograma médio salvo em: {OUTPUT_CSV}')
