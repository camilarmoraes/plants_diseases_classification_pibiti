# Etapa 4 — Treino e Avaliação de Modelos

## Objetivo

Treinamento e avaliação sistemática de **6 arquiteturas de redes neurais** diferentes no dataset Plant Pathology 2020 (4 classes de doenças em folhas de macieira). Para cada modelo, o pipeline inclui:

1. Treinamento com **Early Stopping** e **Model Checkpoint** (salva o melhor modelo)
2. Conversão para **TFLite** e **TFLite Quantizado** (otimizado)
3. Inferência comparativa: TF vs TFLite vs TFLite Quantizado
4. Cálculo de **acurácia** em cada formato
5. Exportação dos resultados em CSV

## Modelos Avaliados

| Script | Arquitetura | Transfer Learning | Input Size |
|--------|-------------|-------------------|------------|
| `modelo_simples.py` | Conv2D simples (1 camada) | ❌ | 200×200 |
| `cnn.py` | CNN (4 camadas Conv2D) | ❌ | 200×200 |
| `mobilenet.py` | MobileNet V1 | ✅ ImageNet | 224×224 |
| `mobilenetv2.py` | MobileNet V2 | ✅ ImageNet | 224×224 |
| `inceptionv3.py` | Inception V3 | ✅ ImageNet | 224×224 |
| `nasnetmobile.py` | NASNet Mobile | ✅ ImageNet | 224×224 |

## Arquitetura dos Modelos com Transfer Learning

Para os modelos pré-treinados (MobileNet, MobileNetV2, InceptionV3, NASNetMobile):
- Base model carregado com pesos ImageNet (`include_top=False`)
- Camadas adicionadas: `Flatten → Dense(512, relu) → Dense(4, softmax)`
- Fine-tuning parcial (últimas camadas descongeladas)
- Callbacks: EarlyStopping (patience=35) + ModelCheckpoint

## Resultados

Cada modelo gera:
- Modelo TensorFlow (`.h5`)
- Modelo TFLite (`.tflite`)
- Modelo TFLite Quantizado (`.tflite`)
- CSV com predições comparativas dos 3 formatos
- Arquivo de texto com acurácia TFLite

## Dependências

- TensorFlow 2.x
- Pandas, NumPy
- Keras Preprocessing
