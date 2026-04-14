# PIBITI — Avaliação de Redes Neurais Profundas no Reconhecimento de Doenças e Pragas em Plantas em Dispositivos Portáteis

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Sobre o Projeto

Projeto de pesquisa do **PIBITI** (Programa Institucional de Bolsas de Iniciação em Desenvolvimento Tecnológico e Inovação) que aplica **visão computacional** utilizando dispositivo de limitado poder computacional — **Raspberry Pi 4B** — para o reconhecimento de doenças e pragas em plantas.

### Objetivos

- Treinar e avaliar múltiplas arquiteturas de redes neurais profundas (CNN, MobileNet, MobileNetV2, InceptionV3, NASNetMobile) na classificação de doenças em folhas de macieira
- Converter modelos para **TensorFlow Lite** (TFLite) e **TFLite Quantizado** para execução em dispositivos com recursos limitados
- Investigar técnicas de **pré-processamento com OpenCV** (histogramas, filtros morfológicos, segmentação por cor) para melhorar a acurácia
- Realizar inferência em tempo real no **Raspberry Pi 4B**, opcionalmente com acelerador **Coral Edge TPU**

### Dataset

[Plant Pathology 2020 — FGVC7](https://www.kaggle.com/c/plant-pathology-2020-fwgp) (Kaggle)

| Classe | Descrição |
|--------|-----------|
| `healthy` | Folhas saudáveis |
| `multiple_diseases` | Múltiplas doenças |
| `rust` | Ferrugem |
| `scab` | Sarna |

## Estrutura do Repositório

Este repositório organiza os scripts do projeto em **7 etapas** sequenciais, cada uma representando uma fase da pesquisa:

```
pibiti-steps-scripts/
│
├── etapa-01-entendimentos-iniciais/        # Primeiros experimentos com TFLite Model Maker
├── etapa-02-treinamentos-especificos/      # Treinamentos com datasets clássicos (MNIST, CIFAR-10, etc.)
├── etapa-03-dataset-escolhido/             # Aplicação no dataset Plant Pathology 2020
├── etapa-04-treino-avaliacao-modelos/       # Treino comparativo de 6 arquiteturas
├── etapa-05-histogramas-preprocessamento/   # Análise de histogramas e pré-processamento OpenCV
├── etapa-06-filtragem-especifica/           # Filtros de cor para isolamento de sintomas
└── etapa-07-inferencia-dispositivos-portateis/  # Inferência TFLite no Raspberry Pi
```

Cada etapa possui um `README.md` com descrição detalhada dos scripts, dependências e instruções de uso.

## Etapas do Projeto

### [Etapa 1 — Entendimentos Iniciais](etapa-01-entendimentos-iniciais/)
Primeiros experimentos com **TFLite Model Maker** usando datasets de flores e objetos escolares. Aprendizado do pipeline: dados → treino → avaliação → exportação TFLite.

### [Etapa 2 — Treinamentos Específicos](etapa-02-treinamentos-especificos/)
Aprofundamento com **datasets clássicos** (MNIST, Fashion MNIST, CIFAR-10, Beans, Rock Paper Scissors). Técnicas de split de dados, K-Fold Cross Validation, e automação de avaliação de múltiplos modelos.

### [Etapa 3 — Dataset Escolhido (Plant Pathology)](etapa-03-dataset-escolhido/)
Primeiros treinos no **dataset final** do projeto. CNN personalizada e DenseNet121 com transfer learning. Inferência comparativa TF vs TFLite vs TFLite quantizado.

### [Etapa 4 — Treino e Avaliação de Modelos](etapa-04-treino-avaliacao-modelos/)
Avaliação sistemática de **6 arquiteturas**: Modelo Simples, CNN, MobileNet, MobileNetV2, InceptionV3 e NASNetMobile. Para cada modelo: treino, conversão TFLite, inferência e cálculo de acurácia.

### [Etapa 5 — Histogramas e Pré-Processamento](etapa-05-histogramas-preprocessamento/)
Análise de histogramas RGB por classe com **OpenCV**. Aplicação de filtros morfológicos, segmentação por cor HSV, e treinamento com imagens pré-processadas (concatenadas).

### [Etapa 6 — Filtragem Específica de Doenças](etapa-06-filtragem-especifica/)
Filtros de cor dedicados para isolar sintomas de **ferrugem (rust)** e **sarna (scab)** com base nos histogramas analisados.

### [Etapa 7 — Inferência em Dispositivos Portáteis](etapa-07-inferencia-dispositivos-portateis/)
Scripts de inferência otimizados para execução no **Raspberry Pi 4B**. Suporte a imagem única ou pasta de imagens, medição de latência, cálculo de acurácia, e compatibilidade com **Coral Edge TPU**.

## Tecnologias Utilizadas

- **TensorFlow 2.x** — Treinamento e conversão de modelos
- **TensorFlow Lite** — Inferência em dispositivos embarcados
- **OpenCV** — Pré-processamento de imagens
- **scikit-learn** — Validação cruzada e split de dados
- **PyCoral** — Inferência com Coral Edge TPU
- **Raspberry Pi 4B** — Dispositivo de inferência
- **Coral USB Accelerator** — Acelerador de inferência (Edge TPU)

## Hardware

| Componente | Especificação |
|------------|---------------|
| Dispositivo | Raspberry Pi 4 Model B |
| RAM | 4 GB |
| Acelerador | Coral USB Accelerator (opcional) |
| SO | Raspberry Pi OS |

## Como Usar

### Requisitos
```bash
pip install tensorflow numpy pillow pandas opencv-python matplotlib scikit-learn
```

### Inferência Rápida (Etapa 7)
```bash
cd etapa-07-inferencia-dispositivos-portateis/

# Classificar uma única imagem
python label_image.py -m modelo.tflite -l plants_labels.txt -i imagem.jpg

# Classificar uma pasta de imagens
python label_image.py -m modelo.tflite -l plants_labels.txt -i ./imagens/

# Inferência em lote com acurácia
python batch_inference.py -m modelo.tflite -l plants_labels.txt -i ./imagens/ \
    --ground_truth test.csv --output resultados.csv
```

## Licença

Scripts de inferência TFLite baseados em exemplos do TensorFlow (Apache 2.0).
