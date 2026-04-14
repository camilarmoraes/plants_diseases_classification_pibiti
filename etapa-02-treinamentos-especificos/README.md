# Etapa 2 — Treinamentos Específicos

## Objetivo

Aprofundamento no treinamento de redes neurais com **datasets clássicos** de machine learning, explorando diferentes arquiteturas, técnicas de split de dados, validação cruzada (K-Fold), conversão para TFLite, e automação de avaliação e predição de múltiplos modelos.

## Estrutura

### `01_codigos_treinamento/` — Treinamentos com Datasets Clássicos

| Script | Dataset | Descrição |
|--------|---------|-----------|
| `mnist.py` | MNIST (dígitos) | Modelo denso simples, treinamento com 15 epochs, plot de accuracy/val_accuracy |
| `fashion_mnist.py` | Fashion MNIST | Modelo denso, conversão para TFLite com quantização |
| `cifar10.py` | CIFAR-10 | Imagens coloridas 32x32, classificação de 10 classes de objetos |
| `beans.py` | Beans (folhas de feijão) | CNN com dataset de doenças em folhas de feijão (3 classes) |
| `pedra_papel_tesoura.py` | Rock Paper Scissors | CNN com 4 camadas convolucionais e data augmentation |
| `testando_mnist.py` | MNIST | Teste rápido com Dropout e plot de accuracy |
| `teste_de_camadas.py` | Fashion MNIST | Teste com Conv2D + MaxPooling para estudar efeito de camadas convolucionais |

### `02_split_dados/` — Divisão de Dados

| Script | Descrição |
|--------|-----------|
| `salvando_carregando.py` | Carrega modelo salvo (.h5) e avalia com splits aleatórios usando `train_test_split` |
| `test_train.py` | Concatena treino+teste, faz split 80/10/10, treina CNN e avalia em cada subset |

### `03_automacao_predicoes/` — Automação e Inferência

| Script | Descrição |
|--------|-----------|
| `programa.py` | Carrega 10 modelos salvos, avalia cada um, seleciona o melhor, converte para TFLite e faz inferência |
| `programaModularizado.py` | Versão modularizada do `programa.py` com funções: `modelEvaluate()`, `modelPredict()`, `convertTFLITE()`, `inferenceTFLITE()` |
| `programaTFLite.py` | Pipeline completo: avalia modelos, converte para TFLite + TFLite quantizado, e compara predições |
| `cross_validation.py` | K-Fold Cross Validation (10 folds) com CNN no Fashion MNIST |
| `kfold.py` | Exemplo didático de uso do `KFold` do scikit-learn |

## Dependências

- TensorFlow 2.x
- scikit-learn (para `train_test_split` e `KFold`)
- Matplotlib
- NumPy
