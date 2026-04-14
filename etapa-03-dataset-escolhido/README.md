# Etapa 3 — Dataset Escolhido (Plant Pathology 2020)

## Objetivo

Aplicação dos conhecimentos das etapas anteriores no **dataset final do projeto**: o [Plant Pathology 2020](https://www.kaggle.com/c/plant-pathology-2020-fwgp) do Kaggle. Este dataset contém imagens de folhas de macieira com 4 classes:

- **healthy** — folhas saudáveis
- **multiple_diseases** — múltiplas doenças
- **rust** — ferrugem
- **scab** — sarna

Nesta etapa, foram realizados os primeiros treinos com o dataset real, conversão para TFLite e TFLite otimizado (quantizado), e inferência comparativa entre os três formatos de modelo.

## Scripts

### `treino_e_inferencia_tf.py`
Pipeline completo com uma **CNN personalizada** (4 camadas Conv2D):
- Treinamento com 100 epochs
- Exportação do modelo (`.h5`)
- Inferência com modelo TensorFlow completo
- Inferência com TFLite e TFLite otimizado (quantizado)
- Exportação dos resultados em CSV

### `densenet121_treino_inferencia.py`
Mesmo pipeline utilizando **DenseNet121** com transfer learning (pesos ImageNet):
- Fine-tuning das últimas camadas
- Camada densa de 512 neurônios + saída softmax de 4 classes
- Inferência TFLite e exportação de resultados

### `modelo_tflite.py`
Script utilitário para **converter modelos** `.h5` para:
- TFLite padrão (`.tflite`)
- TFLite otimizado/quantizado (`.tflite` com `Optimize.DEFAULT`)

### `inferencia_tflite.py`
Script para **inferência isolada** com modelos TFLite já convertidos:
- Carrega imagens de teste
- Redimensiona tensores de entrada/saída
- Gera DataFrame com predições

## Dependências

- TensorFlow 2.x
- Pandas
- NumPy
- Keras Preprocessing
