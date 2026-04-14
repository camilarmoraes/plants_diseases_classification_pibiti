# Etapa 5 — Estudo de Histogramas e Pré-Processamento com OpenCV

## Objetivo

Investigação aprofundada do **pré-processamento de imagens** como forma de melhorar o reconhecimento de doenças em plantas. Esta etapa combina:

1. **Análise de histogramas** — Geração de histogramas RGB para cada classe (healthy, rust, scab, multiple_diseases) para entender a distribuição de cores de cada doença
2. **Filtragem por faixas de cor** — Aplicação de filtros baseados em histograma para isolar regiões de interesse (sintomas da doença)
3. **Concatenação de imagens filtradas** — Criação de imagens concatenadas (original + filtros) para treinar com múltiplas representações
4. **Treinamento com imagens pré-processadas** — Rede neural treinada com imagens modificadas para avaliar impacto do pré-processamento

## Estrutura

### `opencv/` — Scripts de Análise e Pré-Processamento

| Script | Descrição |
|--------|-----------|
| `programa_histogramas.py` | Gera histogramas RGB para **todas as imagens** de cada classe (healthy, scab, rust, multiple_diseases). Salva figuras e tabelas CSV com distribuição de frequência de cada canal |
| `media_histograma.py` | Calcula o **histograma médio** por classe a partir dos CSVs individuais gerados pelo script anterior |
| `modificando_dataset.py` | Gera versões do dataset com **canais de cor removidos** (No_Blue, No_Green, No_Red, etc.) para estudar a importância de cada canal |
| `filtros_morfologicos.py` | Aplica **filtros morfológicos** (erosão, dilatação, opening, closing, gradient, top hat, black hat) e **segmentação HSV inRange** em lote no dataset |
| `histograma1.py` | Análise de histograma RGB e HSV de uma única imagem |
| `histograma2.py` | Equalização de histograma e filtro Gaussian blur com mask HSV |
| `manipulando_histograma.py` | Corte de histograma por faixas de pixel (isolamento de intensidades específicas por canal), divisão da imagem em quadrantes e cálculo de intensidade média |

### Scripts de Filtragem por Histograma e Treinamento

| Script | Descrição |
|--------|-----------|
| `uso_filtro.py` | Aplica filtros de corte por canal (R, G, B) em uma imagem, com parâmetros ajustáveis por classe (Scab, Rust, Multiple Diseases, Healthy) |
| `filtros_concatenacao.py` | Aplica o filtro Scab (corte de canais B, G, R) em lote em todas as imagens do dataset |
| `concatena_imagem.py` | Concatena horizontalmente as imagens filtradas de cada classe, criando imagens compostas (Healthy+Multiple_Diseases+Rust+Scab) |
| `treinamento.py` | Treina uma **CNN com data augmentation** usando as imagens concatenadas (400×1600). Inclui Early Stopping, Model Checkpoint, conversão TFLite e cálculo de acurácia |

## Fluxo de Trabalho

```
Imagens Originais
    ↓
[programa_histogramas.py] → Histogramas RGB por classe
    ↓
[media_histograma.py] → Histograma médio → Define parâmetros de corte
    ↓
[uso_filtro.py / filtros_concatenacao.py] → Imagens filtradas por classe
    ↓
[concatena_imagem.py] → Imagens concatenadas (4 filtros lado a lado)
    ↓
[treinamento.py] → CNN treinada com imagens pré-processadas
```

## Dependências

- OpenCV (`cv2`)
- Matplotlib
- Pandas, NumPy
- TensorFlow 2.x
