# Etapa 1 — Entendimentos Iniciais

## Objetivo

Primeiros experimentos com **TensorFlow** e **TFLite Model Maker** para aprender os fundamentos de classificação de imagens com redes neurais. Nesta etapa, foram realizados testes com datasets didáticos (flores, objetos do dia a dia) para compreender o pipeline completo: carregamento de dados, treinamento, avaliação, exportação e quantização de modelos.

## Scripts

### `flores.py`
Classificação de imagens de flores utilizando o **TFLite Model Maker**. Esse script faz o download automático do dataset `flower_photos` e realiza:
- Treinamento com o Model Maker (transfer learning)
- Avaliação com dados de teste
- Exportação para TFLite (`.tflite`)
- Quantização float16
- Teste com diferentes arquiteturas (EfficientNet, MobileNet V2, Inception V3)

### `objetos.py`
Classificação de objetos de uso escolar (borracha, caneta, clip, grafite, lápis, post-it) com imagens capturadas manualmente. Segue o mesmo pipeline do `flores.py`, mas com um **dataset personalizado** de 6 classes. Testa diferentes quantidades de epochs e modelos (MobileNet V2).

### `labels.txt`
Arquivo contendo os rótulos das 6 classes de objetos utilizados no treinamento: `Borracha`, `Caneta`, `Clip`, `Grafite`, `Lapis`, `Post-it`.

## Dependências

- TensorFlow 2.x
- TFLite Model Maker
- Matplotlib
- NumPy
