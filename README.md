# Compactação do Algoritmo de Comparação de Strings do Snort

Implementação CUDA do algoritmo Aho-Corasick com STT (State Transition Table) compactada, baseada no artigo:

> **"Compactação do Algoritmo de Comparação de Strings do Snort para o uso na Memória Compartilhada de GPUs"**
> 
> José Bonifácio da Silva Júnior, Edward David Moreno, Ricardo Ferreira dos Santos
> 
> WSCAD 2017 - XVIII Simpósio em Sistemas Computacionais de Alto Desempenho

**Disciplina**: TN741 - Computação de Alto Desempenho - UFRRJ

## Objetivo

Comparar o desempenho de diferentes abordagens de memória GPU para o algoritmo Aho-Corasick:
- **Memória Global**: STT compactada em memória global
- **Memória Compartilhada**: STT compactada com cache híbrido em shared memory

O artigo demonstra que a compactação da STT permite seu armazenamento na memória compartilhada da GPU, eliminando a latência de acesso à memória global. Nossa implementação suporta **495 padrões Snort** (2830 estados), alcançando **compressão de 98.7%** (2830 KB → 36 KB).

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

## Instalação de Dependências no Ubuntu

Siga os passos abaixo em um sistema Ubuntu (18.04/20.04/22.04). As instruções instalam ferramentas de compilação, CMake, Python e bibliotecas para gerar os gráficos (`pandas`, `matplotlib`).

- Atualize a lista de pacotes e instale ferramentas básicas:

```bash
sudo apt update
sudo apt install -y build-essential cmake git
```

- Instale o Python e utilitários (apt):

```bash
sudo apt install -y python3 python3-pip python3-venv
```

- Opções para instalar `pandas` e `matplotlib`:

- Via `apt` (rápido, usa pacotes do sistema):

```bash
sudo apt install -y python3-pandas python3-matplotlib
```

- Via `pip` dentro de um ambiente virtual (recomendado):

```bash
# criar e ativar virtualenv
python3 -m venv .venv
source .venv/bin/activate

# atualizar pip e instalar dependências Python
pip install --upgrade pip
pip install pandas matplotlib
```

- CUDA Toolkit 12.x:

Para a versão 12.x do CUDA recomendamos seguir as instruções oficiais da NVIDIA para garantir compatibilidade com seus drivers e GPU. Em algumas distribuições você pode instalar um pacote genérico via apt (`nvidia-cuda-toolkit`), porém ele pode não fornecer a versão 12.x mais recente:

```bash
# alternativa (pode não ser CUDA 12.x):
sudo apt install -y nvidia-cuda-toolkit
```

Consulte a página de downloads da NVIDIA para obter o instalador/ppa adequado à sua distribuição e versão do CUDA.

Observações:

- Recomenda-se usar um `virtualenv` para instalar `pandas`/`matplotlib` via `pip` quando precisar de versões mais recentes ou isoladas do sistema.
- Verifique se os drivers NVIDIA estão instalados e funcionando antes de instalar o CUDA Toolkit.
- Se você pretende compilar os exemplos CUDA neste repositório, confirme que o `nvcc` está acessível no `PATH` após a instalação do CUDA.

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

Testado com **495 padrões Snort** (2830 estados no autômato):

| Tamanho | Serial | GPU Global | GPU Shared | Ganho Shared vs Global |
|---------|--------|------------|------------|------------------------|
| 10 MB   | 81 ms  | 1.46 ms (56x) | 4.08 ms (20x) | -64% (Global melhor) |
| 100 MB  | 817 ms | 18.65 ms (44x) | 14.96 ms (55x) | **+20%** |
| 500 MB  | 4144 ms | 83.59 ms (50x) | 62.49 ms (66x) | **+25%** |
| 1 GB    | ~8500 ms | ~150 ms (~57x) | ~120 ms (~71x) | **~25%** |

**Crossover Point**: ~100 MB (acima disso, Shared Memory é mais eficiente)

Veja [ANALYSIS.md](ANALYSIS.md) para análise detalhada.

## Ambiente de Teste

- **GPU**: NVIDIA GeForce RTX 4060 Ti (34 SMs, 4352 CUDA Cores, 8 GB GDDR6)
- **CPU**: AMD Ryzen 5 5500 (6 cores / 12 threads)
- **RAM**: 32 GB DDR4 3200MHz
- **SO**: Linux (WSL2)

## Autores

- Implementação: Thiago Carvalho
- Disciplina: TN741 - Computação de Alto Desempenho
- Instituição: UFRRJ - Universidade Federal Rural do Rio de Janeiro
