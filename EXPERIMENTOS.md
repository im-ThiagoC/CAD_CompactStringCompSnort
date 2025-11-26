# Sistema de Análise de Performance - Aho-Corasick GPU

## 📊 Descrição

Este sistema executa experimentos de escalabilidade do algoritmo Aho-Corasick implementado em CUDA, realizando **5 iterações** de cada teste e calculando a **média dos resultados** para garantir confiabilidade estatística.

## 🚀 Uso Rápido

### Executar todos os experimentos e gerar gráficos:

```bash
./run_experiments.sh
```

Este script irá:
1. ✅ Executar experimentos de escalabilidade (1 KB até 1 GB)
2. ✅ Realizar 5 iterações de cada experimento
3. ✅ Calcular médias e salvar em CSV
4. ✅ Gerar gráficos de análise

## 📈 Gráficos Gerados

### 1. **Speedup Analysis** (`results/speedup_analysis.png`)
- Comparação de speedup teórico (Lei de Amdahl) vs alcançado
- Eficiência da implementação GPU (% do teórico)
- Gráficos em escala log-log

### 2. **Throughput Analysis** (`results/throughput_analysis.png`)
- Throughput em Mcps (Milhões de caracteres por segundo)
- Comparação entre Serial CPU, GPU Global e GPU Shared Memory

### 3. **Execution Time** (`results/execution_time.png`)
- Tempo de execução por tamanho de entrada
- Gráfico em escala log-log

## 📁 Arquivos de Saída

```
results/
├── experiment_1kb.csv          # Resultados para 1 KB
├── experiment_10kb.csv         # Resultados para 10 KB
├── experiment_100kb.csv        # ... (múltiplos tamanhos)
├── experiment_1024kb.csv       # Resultados para 1 MB
├── experiment_1048576kb.csv    # Resultados para 1 GB
├── summary_results.csv         # Resumo consolidado
├── speedup_analysis.png        # Gráfico de speedup
├── throughput_analysis.png     # Gráfico de throughput
└── execution_time.png          # Gráfico de tempo de execução
```

## 📝 Formato do CSV

Cada arquivo CSV contém:

```csv
Method,Time(ms),Speedup,Throughput(Mcps),Matches
Serial_CPU,7.86,1.00,133.43,8850
GPU_Global,2.64,2.97,415.03,8850
GPU_Shared_Compact,0.42,18.66,2672.91,8850
```

**Colunas:**
- `Method`: Algoritmo utilizado (Serial_CPU, GPU_Global, GPU_Shared_Compact)
- `Time(ms)`: Tempo médio de execução do kernel (5 iterações)
- `Speedup`: Aceleração vs CPU Serial
- `Throughput(Mcps)`: Throughput em milhões de caracteres por segundo
- `Matches`: Número de padrões encontrados

## 🔧 Dependências

### Compilação (C/CUDA):
```bash
# CUDA Toolkit
sudo apt install nvidia-cuda-toolkit

# CMake
sudo apt install cmake
```

### Análise Python:
```bash
# Instalar dependências
pip3 install pandas matplotlib numpy
```

## 🏗️ Compilação Manual

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## 📊 Executar Apenas Gráficos

Se você já tem os arquivos CSV, pode gerar os gráficos diretamente:

```bash
python3 plot_results.py
```

## 🎯 Metodologia

### Múltiplas Iterações
- Cada experimento executa **5 iterações**
- Calcula-se a **média** dos tempos de execução
- Reduz variabilidade e melhora confiabilidade

### Speedup Teórico
O speedup teórico é calculado usando a **Lei de Amdahl**:

```
Speedup = 1 / (S + (1-S)/P)
```

Onde:
- `S` = fração serial (overhead)
- `P` = número de cores paralelos (4352 para RTX 4060 Ti)

**Frações seriais estimadas:**
- Datasets < 10 KB: 50% (alto overhead de inicialização)
- Datasets < 100 KB: 20%
- Datasets < 1 MB: 10%
- Datasets ≥ 1 MB: 5%

### Métodos Comparados

1. **Serial_CPU**: Implementação serial em C
2. **GPU_Global**: GPU usando memória global
3. **GPU_Shared_Compact**: GPU usando memória compartilhada com STT compactada

## 📐 Análise de Resultados

### Speedup Esperado
Para a RTX 4060 Ti (4352 cores):
- **Pequenos datasets** (< 100 KB): 2-5x (limitado por overhead)
- **Médios datasets** (1-10 MB): 10-25x
- **Grandes datasets** (> 100 MB): 15-30x (limitado por memória bandwidth)

### Eficiência
A eficiência é calculada como:
```
Eficiência = (Speedup Alcançado / Speedup Teórico) × 100%
```

Valores > 80% indicam implementação muito eficiente.

## 🐛 Troubleshooting

### Erro: "out of memory"
Reduza o tamanho máximo dos experimentos editando `src/main.cu`:
```c
size_t test_sizes[] = {1024, 10240, 102400, 1048576}; // até 1 MB apenas
```

### Erro: "Python dependencies not found"
```bash
pip3 install pandas matplotlib numpy
```

### Gráficos não aparecem
Os gráficos são salvos em `results/`. Verifique a pasta:
```bash
ls -lh results/*.png
```

## 📊 Exemplo de Saída

```
============================================================
RESUMO DOS RESULTADOS (MÉDIAS)
============================================================

📊 Tamanho: 1024 KB
Método                         Tempo (ms)      Speedup      Throughput (Mcps)
--------------------------------------------------------------------------------
Serial_CPU                     7.86            1.00         133.43
GPU_Global                     2.64            2.97         415.03
GPU_Shared_Compact             0.42            18.66        2672.91
```

## 📚 Referências

- Lei de Amdahl: [Wikipedia](https://en.wikipedia.org/wiki/Amdahl%27s_law)
- Algoritmo Aho-Corasick: [Paper Original](https://dl.acm.org/doi/10.1145/360825.360855)
- CUDA Programming Guide: [NVIDIA Docs](https://docs.nvidia.com/cuda/)

## 📝 Notas

- Os resultados podem variar entre execuções devido a fatores externos (carga do sistema, temperatura da GPU, etc.)
- O sistema de 5 iterações ajuda a estabilizar os resultados
- Para análises científicas, considere aumentar o número de iterações para 10-20

## 🎓 Trabalho Acadêmico

**Disciplina:** TN741 - Computação de Alto Desempenho  
**Instituição:** UFRRJ (Universidade Federal Rural do Rio de Janeiro)  
**GPU Utilizada:** NVIDIA GeForce RTX 4060 Ti (8.9 Compute Capability)
