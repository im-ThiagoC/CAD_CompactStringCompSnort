# Análise dos Resultados

## Resumo Executivo

A implementação demonstra que a **memória compartilhada com STT compactada** é mais eficiente que a memória global para textos grandes (≥100 MB), com ganhos de até **25%** no tempo de execução.

## Configuração de Teste

- **GPU**: NVIDIA GeForce RTX 4060 Ti
  - 34 Multiprocessors, 4352 CUDA Cores
  - 48 KB Shared Memory por bloco
  - 8 GB GDDR6
- **CPU**: AMD Ryzen (baseline serial)
- **RAM**: 32GB 3200Mhz DDR4 (2x16GB)
- **Padrões**: 495 assinaturas tipo Snort IDS (2830 estados no autômato)
- **STT Compactada**: 36 KB (vs 2830 KB original) - **compressão de 98.7%**

## Estratégia de Implementação

### Kernel Híbrido de Shared Memory

Para suportar autômatos grandes (>1024 estados), implementamos uma **abordagem híbrida**:

1. **Shared Memory Cache**:
   - `est0[256]`: Lookup direto do estado 0 (mais acessado)
   - `failure_cache[1024]`: Cache de failure links
   - `output_cache[1024]`: Cache de contagens de output

2. **Global Memory**:
   - `VI`, `NE`, `VE`, `VS`: Estrutura completa da STT compactada

Esta abordagem permite:
- Suportar qualquer tamanho de autômato
- Manter benefícios de shared memory para dados críticos
- Evitar limite de 48KB de shared memory

### Block Sizes

Seguindo a abordagem do artigo original:

- **GPU Global**: `BLOCK_SIZE = 128` - Blocos menores, eficiente com cache L2
- **GPU Shared**: `BLOCK_SIZE = 1024` - Blocos maiores para melhor ocupância

## Resultados por Tamanho (495 padrões)

### Textos Pequenos (≤10 MB)

| Tamanho | Serial | GPU Global | GPU Shared | Melhor |
|---------|--------|------------|------------|--------|
| 1 KB    | 0.01 ms | 0.15 ms | 0.14 ms | Serial |
| 100 KB  | 0.80 ms | 0.22 ms (4x) | 0.50 ms (2x) | Global |
| 1 MB    | 8.01 ms | 0.37 ms (22x) | 2.76 ms (3x) | Global |
| 10 MB   | 81 ms | 1.46 ms (56x) | 4.08 ms (20x) | Global |

**Análise**: Para textos pequenos com automatos grandes:
- O overhead do cache híbrido não compensa
- Cache L2 da RTX 4060 Ti (32 MB) é muito eficiente

### Textos Grandes (≥50 MB)

| Tamanho | Serial | GPU Global | GPU Shared | Ganho Shared |
|---------|--------|------------|------------|--------------|
| 50 MB   | 406 ms | 6.87 ms (59x) | 8.51 ms (48x) | -24% |
| 100 MB  | 817 ms | 18.65 ms (44x) | 14.96 ms (55x) | **+20%** |
| 500 MB  | 4144 ms | 83.59 ms (50x) | 62.49 ms (66x) | **+25%** |
| 1 GB    | ~8500 ms | ~150 ms (~57x) | ~120 ms (~71x) | **~25%** |

**Análise**: Para textos ≥100 MB, a Shared Memory vence porque:
1. O custo do cache híbrido é amortizado por mais acessos
2. `est0` em shared memory acelera significativamente (estado 0 é ~50% dos acessos)
3. `failure_cache` e `output_cache` reduzem acessos à global memory

## Throughput Máximo

| Versão | Throughput Máximo | Alcançado em |
|--------|-------------------|--------------|
| Serial | ~130 Mcps | Qualquer |
| GPU Global | ~7.8 Gcps | 10-50 MB |
| GPU Shared | ~8.4 Gcps | 500 MB |

## Comparação: 69 vs 495 Padrões

| Métrica | 69 Padrões | 495 Padrões |
|---------|------------|-------------|
| Estados | 464 | 2830 |
| STT Original | 464 KB | 2830 KB |
| STT Compactada | 6 KB | 36 KB |
| Crossover Point | 10 MB | 100 MB |
| Ganho Máximo Shared | +37% | +25% |

Com mais padrões, o crossover point aumenta devido ao maior overhead do cache híbrido.

## Conclusões

1. **Compactação é essencial**: Reduz STT em 98.7% (2830 KB → 36 KB)

2. **Kernel Híbrido funciona**: Suporta automatos de qualquer tamanho

3. **Shared Memory vence para textos muito grandes**: Ganhos de 20-25% para ≥100 MB

4. **Crossover point depende do tamanho do autômato**:
   - 69 padrões (464 estados): ~10 MB
   - 495 padrões (2830 estados): ~100 MB

5. **Speedup máximo**: 66x vs CPU serial (500 MB com Shared Memory)

## Melhorias Futuras

1. **Carregar STT completa** quando automato cabe na shared memory
2. **Texture memory** para a STT em global memory
3. **Constant memory** para est0 (acesso broadcast)

## Arquivos de Resultado

- `results/speedup_analysis.png` - Gráfico de speedup
- `results/throughput_analysis.png` - Gráfico de throughput  
- `results/execution_time.png` - Tempo de execução
- `results/summary_results.csv` - Dados tabulados
