#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
label_image.py — Inferência TFLite para classificação de imagens de plantas.

Realiza inferência com modelos TFLite em uma única imagem ou em uma pasta
de imagens, exibindo os top-K resultados e o tempo de inferência para cada
imagem processada.

Uso:
    # Inferência em uma única imagem:
    python label_image.py -m modelo.tflite -l labels.txt -i imagem.jpg

    # Inferência em uma pasta de imagens:
    python label_image.py -m modelo.tflite -l labels.txt -i ./pasta_imagens/

    # Com delegate externo (ex: Edge TPU):
    python label_image.py -m modelo.tflite -l labels.txt -i imagem.jpg \
        -e libedgetpu.so.1

Baseado no exemplo original do TensorFlow Authors (Apache 2.0).
"""

import argparse
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


def run_inference(interpreter, image_path: str, input_mean: float,
                  input_std: float, labels: list[str] = None,
                  top_k: int = 5) -> dict:
    """
    Executa inferência em uma única imagem.

    Args:
        interpreter: Interpretador TFLite já inicializado.
        image_path: Caminho para a imagem.
        input_mean: Média para normalização da entrada.
        input_std: Desvio padrão para normalização da entrada.
        labels: Lista de rótulos (opcional).
        top_k: Número de resultados a exibir.

    Returns:
        Dicionário com 'predictions', 'time_ms', 'top_class_index'.
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

    # Executa a inferência
    interpreter.set_tensor(input_details[0]['index'], input_data)

    start_time = time.perf_counter()
    interpreter.invoke()
    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Processa a saída
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)

    # Obtém top-K
    top_k_indices = results.argsort()[-top_k:][::-1]

    return {
        'predictions': results,
        'time_ms': elapsed_ms,
        'top_class_index': top_k_indices[0],
        'top_k_indices': top_k_indices,
        'floating_model': floating_model,
    }


def print_results(result: dict, labels: list[str] = None,
                  image_name: str = ''):
    """Exibe os resultados de inferência formatados."""
    if image_name:
        print(f'\n--- {image_name} ---')

    results = result['predictions']
    floating_model = result['floating_model']

    for idx in result['top_k_indices']:
        score = float(results[idx]) if floating_model else float(results[idx] / 255.0)
        label = labels[idx] if labels and idx < len(labels) else f'class_{idx}'
        print(f'  {score:08.6f}: {label}')

    print(f'  Tempo de inferência: {result["time_ms"]:.3f} ms')


def collect_image_paths(input_path: str) -> list[str]:
    """
    Coleta caminhos de imagem a partir de um arquivo ou diretório.

    Args:
        input_path: Caminho para uma imagem ou diretório.

    Returns:
        Lista de caminhos de imagem.
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


def main():
    parser = argparse.ArgumentParser(
        description='Inferência TFLite para classificação de imagens de plantas.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '-i', '--input', required=True,
        help='Caminho para uma imagem ou pasta de imagens para classificação.')
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
        '-k', '--top_k', default=5, type=int,
        help='Número de resultados top-K a exibir.')
    parser.add_argument(
        '-e', '--ext_delegate', default=None,
        help='Caminho para biblioteca do delegate externo (ex: Edge TPU).')
    parser.add_argument(
        '-o', '--ext_delegate_options', default=None,
        help='Opções do delegate externo. Formato: "opt1: val1; opt2: val2"')

    args = parser.parse_args()

    # Carrega labels
    labels = load_labels(args.label_file) if args.label_file else None

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

    # Executa inferência
    total_time = 0
    print(f'\nProcessando {len(image_paths)} imagem(ns)...')
    print(f'Modelo: {args.model_file}')
    if labels:
        print(f'Labels: {", ".join(labels)}')

    for img_path in image_paths:
        result = run_inference(
            interpreter, img_path,
            args.input_mean, args.input_std,
            labels, args.top_k)
        print_results(result, labels, os.path.basename(img_path))
        total_time += result['time_ms']

    # Resumo
    if len(image_paths) > 1:
        print(f'\n=== Resumo ===')
        print(f'Total de imagens: {len(image_paths)}')
        print(f'Tempo total: {total_time:.3f} ms')
        print(f'Tempo médio por imagem: {total_time / len(image_paths):.3f} ms')


if __name__ == '__main__':
    main()
