# Compactação do Algoritmo de Comparação de Strings do Snort

Implementação CUDA do algoritmo Aho-Corasick com STT (State Transition Table) compactada, baseada no artigo:

> **"Compactação do Algoritmo de Comparação de Strings do Snort"**
> TN741 - Computação de Alto Desempenho - UFRRJ

## Objetivo

Comparar o desempenho de diferentes abordagens de memória GPU para o algoritmo Aho-Corasick:
- **Memória Global**: STT compactada em memória global
- **Memória Compartilhada**: STT compactada carregada em shared memory

O artigo demonstra que a compactação da STT (de ~464KB para ~6KB) permite seu armazenamento completo na memória compartilhada da GPU, eliminando a latência de acesso à memória global.

## Conceitos Chave

### STT Compactada

A STT (State Transition Table) original ocupa `num_estados × 256 × sizeof(int)` bytes. A versão compactada usa 4 vetores:

- **VI** (Vetor de Índices): Offset no VE/VS para cada estado
- **NE** (Número de Entradas): Quantidade de transições por estado  
- **VE** (Vetor de Entradas): Caracteres que causam transição
- **VS** (Vetor de Saídas): Estado destino para cada transição

### Otimização est0

O estado 0 é especial - possui transições para muitos caracteres. Uma tabela de lookup direto `est0[256]` permite acesso O(1) para o estado inicial.

## Compilação e Execução

```bash
# Compilar
mkdir -p build && cd build
cmake .. && make -j4

# Executar teste rápido (1 MB)
./aho_corasick 1

# Experimento completo
./aho_corasick 4

# Executar todos os experimentos e gerar gráficos
cd .. && ./run_experiments.sh
```

## Requisitos

- CUDA Toolkit 12.x
- CMake 3.18+
- GPU NVIDIA (testado em RTX 4060 Ti)
- Python 3 com matplotlib e pandas (para gráficos)

## Estrutura do Projeto

```
├── src/
│   ├── main.cu                 # Programa principal
│   ├── aho_corasick_gpu.cu     # Kernels CUDA
│   ├── aho_corasick_serial.c   # Implementação serial
│   └── utils.cu                # Funções auxiliares
├── include/
│   ├── aho_corasick.h          # Headers do Aho-Corasick
│   ├── config.h                # Configurações
│   └── utils.h                 # Headers utilitários
├── data/
│   └── patterns.txt            # Padrões de busca
├── results/                    # Resultados e gráficos
├── run_experiments.sh          # Script de experimentos
└── plot_results.py             # Geração de gráficos
```

## Resultados

Para textos grandes (≥10 MB), a versão com memória compartilhada é consistentemente mais rápida:

| Tamanho | Global Speedup | Shared Speedup | Ganho Shared |
|---------|----------------|----------------|--------------|
| 10 MB   | 60x            | 95x            | +37%         |
| 50 MB   | 105x           | 136x           | +23%         |
| 500 MB  | 129x           | 158x           | +18%         |
| 1 GB    | 132x           | 157x           | +16%         |

Veja [ANALYSIS.md](ANALYSIS.md) para análise detalhada.

## Autores

- Implementação: Thiago Cardoso
- Disciplina: TN741 - Computação de Alto Desempenho
- Instituição: UFRRJ - Universidade Federal Rural do Rio de Janeiro
