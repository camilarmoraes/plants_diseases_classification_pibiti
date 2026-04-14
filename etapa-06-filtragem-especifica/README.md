# Etapa 6 — Filtragem Específica de Doenças

## Objetivo

Aplicação de **filtros de cor específicos** para isolar os sintomas visuais de cada doença nas imagens de folhas. Com base nos histogramas analisados na Etapa 5, foram definidas faixas de pixel para cada canal (R, G, B) que destacam as regiões das doenças:

- **Rust (Ferrugem)**: tons alaranjados/amarelados → filtro por intensidade alta em R e G
- **Scab (Sarna)**: tons escuros e amarelados → filtro por intensidade moderada com dupla filtragem
- **Multiple Diseases**: combinação de sintomas → filtro por faixas intermediárias
- **Healthy**: remoção para uso como referência

## Scripts

### `filtro_cores_amarelas.py`
Filtra **cores amarelas e alaranjadas** das imagens, isolando pixels com:
- Canal vermelho (R) entre 130–255
- Canal verde (G) entre 0–200

Aplica-se sobre imagens já pré-processadas pela Etapa 5 (com filtro de histograma aplicado). O resultado é uma imagem onde apenas os pixels das regiões doentes são preservados (o restante fica preto).

### `usando_filtros_duplos.py`
Aplica **dois filtros em sequência** para processar todas as imagens do dataset:

1. **Filtro para Rust**: Preserva pixels com R ≥ 130 e G ≤ 200 (tons alaranjados)
2. **Filtro para Scab**: Preserva pixels com G ≥ 120 e R ≥ 120 (tons amarelo-esverdeados)

Processa todas as imagens em lote e salva os resultados nas pastas correspondentes.

## Fluxo de Trabalho

```
Imagens com filtro de histograma (Etapa 5)
    ↓
[filtro_cores_amarelas.py] → Teste com imagem individual (ajuste de parâmetros)
    ↓
[usando_filtros_duplos.py] → Aplicação em lote no dataset (Rust e Scab)
    ↓
Imagens com regiões de doença isoladas
```

## Dependências

- OpenCV (`cv2`)
- NumPy
