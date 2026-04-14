"""
programa_histogramas.py — Geração de histogramas RGB por classe.

Gera histogramas RGB para todas as imagens de cada classe do dataset
(healthy, scab, rust, multiple_diseases). Para cada imagem, calcula
a distribuição de frequência dos canais B, G, R e salva:
  - Figuras dos histogramas (PNG/JPG)
  - Tabelas CSV com valores de frequência por canal

Dataset: Plant Pathology 2020 (4 classes)
"""

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# ─── Configuração ─────────────────────────────────────────────────────────────
# Diretórios das imagens divididas por classe
CLASSES = {
    'healthy': './div_train/healthy/',
    'multiple_diseases': './div_train/multiple_diseases/',
    'rust': './div_train/rust/',
    'scab': './div_train/scab/',
}

# Diretório de saída para histogramas (figuras e tabelas)
OUTPUT_BASE = './Histogramas/'
# ──────────────────────────────────────────────────────────────────────────────


def compute_histogram(img):
    """
    Calcula a distribuição de frequência dos canais B, G, R.

    Args:
        img: Imagem BGR carregada com cv2.imread

    Returns:
        Tupla (resultado_azul, resultado_verde, resultado_vermelho),
        cada uma com 256 valores de frequência.
    """
    canal_azul = []
    canal_verde = []
    canal_vermelho = []

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            canal_azul.append(img[i][j][0])
            canal_verde.append(img[i][j][1])
            canal_vermelho.append(img[i][j][2])

    def count_frequencies(channel):
        unique_vals = sorted(set(channel))
        result = []
        for val in unique_vals:
            result.append(channel.count(val))
        # Preenche até 256 valores
        result.extend([0] * (256 - len(result)))
        return result

    return (count_frequencies(canal_azul),
            count_frequencies(canal_verde),
            count_frequencies(canal_vermelho))


def process_class(class_name, images_dir):
    """Gera histogramas para todas as imagens de uma classe."""
    output_fig = os.path.join(OUTPUT_BASE, class_name, 'Figuras')
    output_csv = os.path.join(OUTPUT_BASE, class_name, 'Tabelas')
    os.makedirs(output_fig, exist_ok=True)
    os.makedirs(output_csv, exist_ok=True)

    for filename in os.listdir(images_dir):
        img = cv2.imread(os.path.join(images_dir, filename))
        if img is None:
            continue

        resultado_azul, resultado_verde, resultado_vermelho = \
            compute_histogram(img)

        # Salva figura do histograma
        plt.plot(resultado_azul, color='blue')
        plt.plot(resultado_verde, color='green')
        plt.plot(resultado_vermelho, color='red')
        plt.savefig(os.path.join(output_fig, filename), format='jpg')
        plt.show()

        # Salva tabela CSV
        df = pd.DataFrame({
            'R': resultado_vermelho,
            'G': resultado_verde,
            'B': resultado_azul
        })
        df.to_csv(os.path.join(output_csv, f'{filename}.csv'))


# ─── Execução ────────────────────────────────────────────────────────────────
for class_name, images_dir in CLASSES.items():
    print(f'Processando classe: {class_name}')
    process_class(class_name, images_dir)
