#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_inference.py — Inferência em lote com TFLite e medição de latência.

Realiza inferência em uma única imagem ou em uma pasta de imagens usando
modelos TFLite. Opcionalmente, carrega um CSV com ground truth para calcular
a acurácia do modelo.

Uso:
    # Inferência em uma pasta de imagens:
    python batch_inference.py -m modelo.tflite -l labels.txt -i ./imagens/

    # Inferência em uma única imagem:
    python batch_inference.py -m modelo.tflite -l labels.txt -i imagem.jpg

    # Com CSV de ground truth para calcular acurácia:
    python batch_inference.py -m modelo.tflite -l labels.txt -i ./imagens/ \
        --ground_truth test.csv --gt_column "Valores por Coluna"

    # Exportar resultados em CSV:
    python batch_inference.py -m modelo.tflite -l labels.txt -i ./imagens/ \
        --output resultados.csv

    # Com delegate externo (ex: Edge TPU):
    python batch_inference.py -m modelo.tflite -l labels.txt -i ./imagens/ \
        -e libedgetpu.so.1

"""

import argparse
import csv
import os
import sys
import time
import glob

import numpy as np
from PIL import Image

# Tenta importar o TFLite runtime standalone (para Raspberry Pi),
# e usa o TensorFlow completo como fallback.
try:
    import tflite_runtime.interpreter as tflite
    USING_TFLITE_RUNTIME = True
except ImportError:
    import tensorflow as tf
    tflite = tf.lite
    USING_TFLITE_RUNTIME = False


# ─── Extensões de imagem suportadas ───────────────────────────────────────────
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')


def load_labels(filename: str) -> list[str]:
    """Carrega rótulos de um arquivo de texto (um rótulo por linha)."""
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines()]


def load_ground_truth(csv_path: str, column_name: str) -> dict:
    """
    Carrega ground truth de um CSV.

    Espera um CSV com coluna de índice (image_id) e uma coluna numérica
    com o índice da classe correta.

    Args:
        csv_path: Caminho para o arquivo CSV.
        column_name: Nome da coluna com os valores de ground truth.

    Returns:
        Dicionário {image_id: class_index}.
    """
    import pandas as pd
    df = pd.read_csv(csv_path, index_col=0)
    return dict(zip(df.index, df[column_name].values))


def create_interpreter(model_path: str, ext_delegate: str = None,
                       ext_delegate_options: dict = None,
                       num_threads: int = None):
    """
    Cria e retorna um interpretador TFLite.

    Args:
        model_path: Caminho para o modelo .tflite.
        ext_delegate: Caminho para a biblioteca do delegate externo (opcional).
        ext_delegate_options: Opções do delegate externo (opcional).
        num_threads: Número de threads para inferência (opcional).

    Returns:
        Interpretador TFLite pronto para uso.
    """
    delegates = None

    if ext_delegate is not None:
        print(f'Carregando delegate externo: {ext_delegate}')
        if USING_TFLITE_RUNTIME:
            delegates = [tflite.load_delegate(ext_delegate,
                                              ext_delegate_options or {})]
        else:
            delegates = [tf.lite.experimental.load_delegate(
                ext_delegate, ext_delegate_options or {})]

    if USING_TFLITE_RUNTIME:
        interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=delegates,
            num_threads=num_threads)
    else:
        interpreter = tflite.Interpreter(
            model_path=model_path,
            experimental_delegates=delegates,
            num_threads=num_threads)

    interpreter.allocate_tensors()
    return interpreter


def run_single_inference(interpreter, image_path: str, input_mean: float,
                         input_std: float) -> dict:
    """
    Executa inferência em uma única imagem.

    Args:
        interpreter: Interpretador TFLite já inicializado.
        image_path: Caminho para a imagem.
        input_mean: Média para normalização da entrada.
        input_std: Desvio padrão para normalização da entrada.

    Returns:
        Dicionário com 'raw_output', 'predicted_class', 'scores', 'time_ms'.
    """
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Verifica tipo do tensor de entrada
    floating_model = input_details[0]['dtype'] == np.float32

    # Obtém dimensões esperadas pelo modelo (NxHxWxC)
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]

    # Carrega e redimensiona a imagem
    img = Image.open(image_path).convert('RGB').resize((width, height))
    input_data = np.expand_dims(img, axis=0)

    # Normalização (para modelos float)
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # Executa a inferência com medição de latência
    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.perf_counter()
    interpreter.invoke()
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Processa a saída
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    # Normaliza scores
    if floating_model:
        scores = results
    else:
        scores = results / 255.0

    predicted_class = int(np.argmax(scores))

    return {
        'raw_output': results,
        'predicted_class': predicted_class,
        'scores': scores,
        'time_ms': elapsed_ms,
        'floating_model': floating_model,
    }


def collect_image_paths(input_path: str) -> list[str]:
    """
    Coleta caminhos de imagem a partir de um arquivo ou diretório.

    Args:
        input_path: Caminho para uma imagem ou diretório.

    Returns:
        Lista ordenada de caminhos de imagem.
    """
    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        images = []
        for ext in SUPPORTED_EXTENSIONS:
            images.extend(glob.glob(os.path.join(input_path, f'*{ext}')))
            images.extend(glob.glob(os.path.join(input_path, f'*{ext.upper()}')))
        images = sorted(set(images))
        if not images:
            print(f'Aviso: Nenhuma imagem encontrada em {input_path}')
        return images

    print(f'Erro: Caminho não encontrado: {input_path}')
    sys.exit(1)


def parse_delegate_options(options_str: str) -> dict:
    """Converte string de opções do delegate para dicionário."""
    options = {}
    if options_str:
        for opt in options_str.split(';'):
            kv = opt.split(':')
            if len(kv) == 2:
                options[kv[0].strip()] = kv[1].strip()
            else:
                raise RuntimeError(f'Erro ao parsear opção do delegate: {opt}')
    return options


def export_results_csv(output_path: str, results: list[dict],
                       labels: list[str] = None):
    """
    Exporta resultados de inferência para CSV.

    Args:
        output_path: Caminho para o arquivo CSV de saída.
        results: Lista de dicionários com resultados.
        labels: Lista de rótulos (opcional).
    """
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Header
        header = ['image', 'predicted_class', 'predicted_label', 'time_ms']
        if labels:
            header.extend([f'score_{l}' for l in labels])
        writer.writerow(header)

        # Rows
        for r in results:
            row = [
                r['image_name'],
                r['predicted_class'],
                r.get('predicted_label', ''),
                f"{r['time_ms']:.3f}",
            ]
            if labels:
                row.extend([f"{s:.6f}" for s in r['scores']])
            writer.writerow(row)

    print(f'Resultados exportados para: {output_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Inferência TFLite em lote com medição de latência.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-i', '--input', required=True,
        help='Caminho para uma imagem ou pasta de imagens.')
    parser.add_argument(
        '-m', '--model_file', required=True,
        help='Caminho para o modelo .tflite.')
    parser.add_argument(
        '-l', '--label_file', default=None,
        help='Caminho para o arquivo de rótulos (um por linha).')
    parser.add_argument(
        '--input_mean', default=127.5, type=float,
        help='Média para normalização da entrada.')
    parser.add_argument(
        '--input_std', default=127.5, type=float,
        help='Desvio padrão para normalização da entrada.')
    parser.add_argument(
        '--num_threads', default=None, type=int,
        help='Número de threads para inferência.')
    parser.add_argument(
        '-e', '--ext_delegate', default=None,
        help='Caminho para biblioteca do delegate externo (ex: Edge TPU).')
    parser.add_argument(
        '-o', '--ext_delegate_options', default=None,
        help='Opções do delegate externo. Formato: "opt1: val1; opt2: val2"')
    parser.add_argument(
        '--ground_truth', default=None,
        help='Caminho para CSV com ground truth (para cálculo de acurácia).')
    parser.add_argument(
        '--gt_column', default='Valores por Coluna',
        help='Nome da coluna de ground truth no CSV.')
    parser.add_argument(
        '--output', default=None,
        help='Caminho para exportar resultados em CSV.')

    args = parser.parse_args()

    # Carrega labels
    labels = load_labels(args.label_file) if args.label_file else None

    # Carrega ground truth (se fornecido)
    ground_truth = None
    if args.ground_truth:
        try:
            ground_truth = load_ground_truth(args.ground_truth, args.gt_column)
            print(f'Ground truth carregado: {len(ground_truth)} entradas')
        except Exception as e:
            print(f'Aviso: Falha ao carregar ground truth: {e}')

    # Parse de opções do delegate
    delegate_opts = parse_delegate_options(args.ext_delegate_options)

    # Cria o interpretador
    interpreter = create_interpreter(
        model_path=args.model_file,
        ext_delegate=args.ext_delegate,
        ext_delegate_options=delegate_opts if delegate_opts else None,
        num_threads=args.num_threads)

    # Coleta imagens
    image_paths = collect_image_paths(args.input)

    if not image_paths:
        print('Nenhuma imagem para processar.')
        sys.exit(0)

    # ─── Executa inferência ───────────────────────────────────────────────
    print(f'\n{"="*60}')
    print(f'Modelo: {args.model_file}')
    if labels:
        print(f'Labels: {", ".join(labels)}')
    print(f'Imagens a processar: {len(image_paths)}')
    print(f'{"="*60}\n')

    all_results = []
    correct = 0
    total_gt = 0

    for img_path in image_paths:
        result = run_single_inference(
            interpreter, img_path, args.input_mean, args.input_std)

        image_name = os.path.basename(img_path)
        image_id = os.path.splitext(image_name)[0]

        predicted_label = ''
        if labels and result['predicted_class'] < len(labels):
            predicted_label = labels[result['predicted_class']]

        # Exibe resultado
        print(f'[{image_name}] '
              f'Predição: {result["predicted_class"]} ({predicted_label}) | '
              f'Latência: {result["time_ms"]:.3f} ms', end='')

        # Compara com ground truth
        if ground_truth and image_id in ground_truth:
            gt_class = int(ground_truth[image_id])
            is_correct = result['predicted_class'] == gt_class
            correct += int(is_correct)
            total_gt += 1
            status = '✓' if is_correct else '✗'
            print(f' | GT: {gt_class} [{status}]', end='')

        print()

        all_results.append({
            'image_name': image_name,
            'image_id': image_id,
            'predicted_class': result['predicted_class'],
            'predicted_label': predicted_label,
            'scores': result['scores'],
            'time_ms': result['time_ms'],
        })

    # ─── Resumo ───────────────────────────────────────────────────────────
    total_time = sum(r['time_ms'] for r in all_results)

    print(f'\n{"="*60}')
    print(f'RESUMO')
    print(f'{"="*60}')
    print(f'Total de imagens processadas: {len(all_results)}')
    print(f'Tempo total de inferência:    {total_time:.3f} ms')
    print(f'Tempo médio por imagem:       {total_time / len(all_results):.3f} ms')

    if total_gt > 0:
        accuracy = correct / total_gt
        errors = total_gt - correct
        print(f'\n--- Acurácia ---')
        print(f'Acertos: {correct}/{total_gt}')
        print(f'Erros:   {errors}/{total_gt}')
        print(f'Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)')

    print(f'{"="*60}')

    # Exporta resultados em CSV (se solicitado)
    if args.output:
        export_results_csv(args.output, all_results, labels)


if __name__ == '__main__':
    main()
