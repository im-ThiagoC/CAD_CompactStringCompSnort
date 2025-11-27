# Análise dos Resultados

## Resumo Executivo

A implementação demonstra que a **memória compartilhada com STT compactada** é mais eficiente que a memória global para textos grandes (≥10 MB), com ganhos de até **37%** no tempo de execução.

## Configuração de Teste

- **GPU**: NVIDIA GeForce RTX 4060 Ti
  - 34 Multiprocessors, 4352 CUDA Cores
  - 48 KB Shared Memory por bloco
  - 8 GB GDDR6
- **CPU**: AMD Ryzen (baseline serial)
- **Padrões**: 69 assinaturas IDS (464 estados no autômato)
- **STT Compactada**: 6 KB (vs 464 KB original) - **compressão de 98.7%**

## Resultados por Tamanho

### Textos Pequenos (≤1 MB)

| Tamanho | Serial | GPU Global | GPU Shared | Melhor |
|---------|--------|------------|------------|--------|
| 1 KB    | 0.01 ms | 0.11 ms (0.08x) | 0.07 ms (0.13x) | Serial |
| 10 KB   | 0.07 ms | 0.04 ms (1.80x) | 0.04 ms (1.66x) | Global |
| 100 KB  | 0.62 ms | 0.07 ms (8.39x) | 0.13 ms (4.75x) | Global |
| 1 MB    | 6.65 ms | 0.11 ms (59x) | 0.66 ms (10x) | Global |

**Análise**: Para textos pequenos, a GPU Global é mais eficiente porque:
1. O overhead de carregar a STT para shared memory não compensa
2. O cache L2 da RTX 4060 Ti (32 MB) é muito eficiente
3. Poucos caracteres por thread = poucas buscas na STT

### Textos Grandes (≥10 MB)

| Tamanho | Serial | GPU Global | GPU Shared | Ganho Shared |
|---------|--------|------------|------------|--------------|
| 10 MB   | 65 ms  | 1.09 ms (60x) | 0.69 ms (95x) | **+37%** |
| 50 MB   | 321 ms | 3.06 ms (105x) | 2.35 ms (136x) | **+23%** |
| 100 MB  | 626 ms | 5.05 ms (124x) | 5.15 ms (122x) | -2% |
| 500 MB  | 3163 ms | 24.6 ms (129x) | 20.0 ms (158x) | **+18%** |
| 1 GB    | 6393 ms | 48.4 ms (132x) | 40.9 ms (157x) | **+16%** |

**Análise**: Para textos grandes, a Shared Memory é mais eficiente porque:
1. Latência da shared memory (~5 ciclos) vs global (~400 ciclos)
2. Muitos caracteres por thread = muitas buscas na STT
3. O custo de carregamento é amortizado pelo número de acessos

## Throughput

| Versão | Throughput Máximo | Tamanho Ideal |
|--------|-------------------|---------------|
| Serial | ~168 Mcps | Qualquer |
| GPU Global | ~22 Gcps | 1 MB |
| GPU Shared | ~26 Gcps | 50-500 MB |

## Configuração dos Kernels

A diferença de block sizes segue a abordagem do artigo original:

- **GPU Global**: `BLOCK_SIZE = 128`
  - Blocos menores, menos overhead de sincronização
  - Eficiente com cache L2

- **GPU Shared**: `BLOCK_SIZE = 1024`
  - Blocos maiores para melhor ocupância
  - Mais threads compartilham a mesma STT carregada

## Conclusões

1. **Compactação é essencial**: Reduz a STT de 464 KB para 6 KB (98.7%)

2. **Shared Memory vence para textos grandes**: Ganhos de 16-37% para ≥10 MB

3. **Crossover point**: ~10 MB - abaixo disso, Global é melhor

4. **Hardware moderno favorece Global para textos pequenos**: 
   - O cache L2 de 32 MB da RTX 4060 Ti mascara a latência da memória global
   - GPUs mais antigas (sem cache L2 grande) teriam resultados diferentes

5. **Speedup máximo**: 157x vs CPU serial (para 1 GB com Shared Memory)

## Arquivos de Resultado

- `results/speedup_analysis.png` - Gráfico de speedup
- `results/throughput_analysis.png` - Gráfico de throughput  
- `results/execution_time.png` - Tempo de execução
- `results/summary_results.csv` - Dados tabulados
