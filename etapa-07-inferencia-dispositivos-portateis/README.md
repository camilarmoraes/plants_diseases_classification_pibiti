# Etapa 7 — Inferência em Dispositivos Portáteis

## Objetivo

Execução de inferência dos modelos treinados em **dispositivos de limitado poder computacional**, especificamente o **Raspberry Pi 4B**, utilizando modelos no formato **TFLite** (TensorFlow Lite) e opcionalmente com acelerador **Coral Edge TPU**.

Esta etapa representa a aplicação final do projeto: classificar imagens de folhas de plantas em tempo real no dispositivo portátil, medindo a **latência** e a **acurácia** dos modelos quantizados.

## Scripts

### `label_image.py` ⭐ (reescrito)

Script de inferência limpo e modularizado. Classifica imagens usando modelos TFLite, exibindo os **top-K resultados** com scores e tempo de inferência.

**Funcionalidades:**
- Aceita **uma imagem** ou **uma pasta de imagens** como entrada
- Todos os caminhos são configuráveis via argumentos CLI (sem hardcode)
- Mede latência de inferência por imagem
- Suporte a delegate externo (Edge TPU)
- Compatível com `tflite_runtime` (Raspberry Pi) e TensorFlow completo

```bash
# Inferência em uma única imagem
python label_image.py -m mobilenet_1.0_quant.tflite -l plants_labels.txt -i Train_2.jpg

# Inferência em uma pasta de imagens
python label_image.py -m mobilenet_1.0_quant.tflite -l plants_labels.txt -i ./imagens/

# Com Edge TPU
python label_image.py -m mobilenet_1.0_quant_edgetpu.tflite -l plants_labels.txt \
    -i ./imagens/ -e libedgetpu.so.1
```

### `batch_inference.py` ⭐ (reescrito)

Script de inferência em lote com medição de latência e cálculo de acurácia. Baseado no script original `inferencia+latencia.py`, mas completamente reescrito.

**Funcionalidades:**
- Aceita **uma imagem** ou **uma pasta de imagens** como entrada
- Calcula **acurácia** quando um CSV de ground truth é fornecido
- Mede **latência total e média** por imagem
- Exporta resultados detalhados em **CSV**
- Suporte a delegate externo (Edge TPU)
- Código modularizado com funções documentadas

```bash
# Inferência em lote
python batch_inference.py -m modelo.tflite -l plants_labels.txt -i ./imagens/

# Com ground truth e exportação de CSV
python batch_inference.py -m modelo.tflite -l plants_labels.txt -i ./imagens/ \
    --ground_truth test.csv --gt_column "Valores por Coluna" --output resultados.csv
```

### `classify_edgetpu.py` (cópia original)

Script do **PyCoral** para classificação com **Coral Edge TPU**. Utiliza a API PyCoral para inferência acelerada por hardware.

```bash
python classify_edgetpu.py -m mobilenet_1.0_quant_edgetpu.tflite \
    -l plants_labels.txt -i imagem.jpg
```

### `plants_labels.txt`

Arquivo de rótulos com as 4 classes do dataset:
```
healthy
multiple_diseases
rust
scab
```

## Argumentos CLI Disponíveis

### `label_image.py`

| Argumento | Descrição | Obrigatório |
|-----------|-----------|-------------|
| `-i, --input` | Caminho para imagem ou pasta de imagens | ✅ |
| `-m, --model_file` | Caminho para o modelo `.tflite` | ✅ |
| `-l, --label_file` | Caminho para arquivo de rótulos | ❌ |
| `-k, --top_k` | Número de resultados top-K | ❌ (default: 5) |
| `--input_mean` | Média para normalização | ❌ (default: 127.5) |
| `--input_std` | Desvio padrão para normalização | ❌ (default: 127.5) |
| `--num_threads` | Número de threads | ❌ |
| `-e, --ext_delegate` | Caminho para delegate externo | ❌ |

### `batch_inference.py`

| Argumento | Descrição | Obrigatório |
|-----------|-----------|-------------|
| `-i, --input` | Caminho para imagem ou pasta de imagens | ✅ |
| `-m, --model_file` | Caminho para o modelo `.tflite` | ✅ |
| `-l, --label_file` | Caminho para arquivo de rótulos | ❌ |
| `--ground_truth` | CSV de ground truth para cálculo de acurácia | ❌ |
| `--gt_column` | Nome da coluna de ground truth | ❌ (default: "Valores por Coluna") |
| `--output` | Caminho para exportar resultados em CSV | ❌ |
| `--input_mean` | Média para normalização | ❌ (default: 127.5) |
| `--input_std` | Desvio padrão para normalização | ❌ (default: 127.5) |
| `--num_threads` | Número de threads | ❌ |
| `-e, --ext_delegate` | Caminho para delegate externo | ❌ |

## Dependências

- TensorFlow Lite Runtime (`tflite_runtime`) **ou** TensorFlow 2.x
- NumPy
- Pillow (PIL)
- Pandas (apenas para `batch_inference.py` com ground truth)
- PyCoral (apenas para `classify_edgetpu.py`)

## Observações

- Os códigos originais de `label_image.py` e `batch_inference.py` são pertencentes ao repositório do google sobre o TFLite. Podendo ser encontrado no seguinte link `https://github.com/google-coral/tflite/tree/master/python/examples/classification/`.`
- Os scripts foram adaptados para o projeto, adicionando funcionalidades como:
    - Aceita uma imagem ou uma pasta de imagens como entrada
    - Todos os caminhos são configuráveis via argumentos CLI (sem hardcode)
    - Mede latência de inferência por imagem
    - Suporte a delegate externo (Edge TPU)
    - Compatível com `tflite_runtime` (Raspberry Pi) e TensorFlow completo
    