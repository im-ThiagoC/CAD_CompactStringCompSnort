# Compactação do Algoritmo de Comparação de Strings do Snort

**Disciplina:** TN741 - Computação de Alto Desempenho  
**Instituição:** UFRRJ - Universidade Federal Rural do Rio de Janeiro  
**Artigo Base:** "Compactação do Algoritmo de Comparação de Strings do Snort para uso na Memória Compartilhada de GPUs" (WSCAD 2017)

## 📋 Sobre o Projeto

Implementação do algoritmo **Aho-Corasick (AC)** para detecção de padrões em sistemas IDS (Intrusion Detection System) como o Snort, com paralelização em GPU usando CUDA.

### Objetivo

Comparar o desempenho de diferentes abordagens de paralelização:
- **Versão Serial (CPU)**
- **GPU com Memória Global**
- **GPU com Memória Compartilhada Compactada**

## 🔧 Tecnologias Utilizadas

- **CUDA**
- **C**
- **CMake 3.18+**
- **GPU NVIDIA** (No meu caso a 4060Ti)

## 📁 Estrutura do Projeto

```
CAD_CompactStringCompSnort/
├── CMakeLists.txt          # Configuração do build
├── README.md               # Esta documentação
├── include/
│   ├── aho_corasick.h     # Interface do algoritmo AC
│   ├── config.h           # Configurações globais
│   └── utils.h            # Funções utilitárias
├── src/
│   ├── main.cu            # Programa principal
│   ├── aho_corasick_serial.cpp  # Implementação serial
│   ├── aho_corasick_gpu.cu      # Implementação GPU
│   └── utils.cu           # Utilitários
├── data/
│   └── patterns.txt       # Padrões de busca (IDS)
├── results/               # Resultados dos experimentos
└── build/                 # Arquivos de compilação
```

## 🚀 Como Compilar e Executar

### Pré-requisitos

1. **CUDA Toolkit** instalado
2. **GPU NVIDIA** compatível
3. **CMake** 3.18 ou superior
4. **GCC/G++** ou compilador compatível

### Compilação (Linux/WSL)

```bash
# Clone ou navegue até o diretório do projeto
cd CAD_CompactStringCompSnort

# Crie o diretório de build
mkdir -p build
cd build

# Configure com CMake
cmake ..

# Compile
make -j$(nproc)

# Execute
./aho_corasick
```

### Compilação (Windows com Visual Studio)

```powershell
# No PowerShell
cd CAD_CompactStringCompSnort
mkdir build
cd build

cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release

# Execute
.\Release\aho_corasick.exe
```

## 📊 Executando Experimentos

O programa oferece 4 opções de experimentos:

1. **Teste Rápido (1 MB)** - Para validação inicial
2. **Experimento Completo** - Testa 1, 10, 50 e 100 MB
3. **Comparação Detalhada** - Foco em 10, 50 e 100 MB
4. **Teste de Escalabilidade** - 1 KB até 1 GB

### Exemplo de Saída

```
╔════════════════════════════════════════════════════════════╗
║  Compactação do Algoritmo de Comparação de Strings (AC)   ║
║  Implementação CUDA - TN741 CAD - UFRRJ                   ║
╚════════════════════════════════════════════════════════════╝

=== Informações da GPU ===
Dispositivos CUDA encontrados: 1

Dispositivo 0: NVIDIA GeForce RTX 4060 Ti
  Compute Capability: 8.9
  Memória Global: 8 GB
  ...

=== MENU DE EXPERIMENTOS ===
1. Teste rápido (1 MB)
2. Experimento completo
3. Comparação detalhada
4. Teste de escalabilidade

Escolha uma opção (1-4): 
```

## 📈 Análise de Resultados

### Sistema de 5 Iterações

O sistema executa **5 iterações** de cada experimento e calcula médias estatísticas para maior precisão:

```bash
# Executar experimentos e gerar gráficos automaticamente
cd build
./run_experiments.sh
```

Os resultados são salvos automaticamente em arquivos CSV no diretório `results/`:

```
results/
├── experiment_1kb.csv         # Resultados por tamanho (média de 5 iterações)
├── experiment_10kb.csv
├── experiment_100kb.csv
├── experiment_1024kb.csv
├── experiment_10240kb.csv
├── experiment_51200kb.csv
├── experiment_102400kb.csv
├── experiment_512000kb.csv
├── experiment_1048576kb.csv
├── summary_results.csv        # Tabela consolidada com eficiência teórica
├── speedup_analysis.png       # Gráfico: Speedup teórico vs alcançado
└── execution_time.png         # Gráfico: Tempo de execução por tamanho
```

### Formato do CSV

```csv
Method,Time(ms),Speedup,Throughput(Mcps),Matches
Serial_CPU,125.50,1.00,8.15,42
GPU_Global,15.20,8.26,80.00,42
GPU_Shared_Compact,8.50,14.76,142.00,42
```

### Métricas Avaliadas

- **Tempo de Execução (ms)**: Tempo total incluindo transferências (média de 5 iterações)
- **Tempo do Kernel (ms)**: Apenas tempo de processamento na GPU (média de 5 iterações)
- **Throughput (Mcps)**: Megacaracteres processados por segundo
- **Speedup**: Ganho em relação à versão serial
- **Eficiência (%)**: Porcentagem do speedup teórico alcançado (Lei de Amdahl)
- **Speedup Teórico**: Calculado usando Lei de Amdahl com fração serial adaptativa

### Geração de Gráficos

#### Opção 1: Script Automatizado (Recomendado)

```bash
cd build
./run_experiments.sh
```

Este script:
1. Limpa resultados anteriores
2. Executa experimento de escalabilidade (5 iterações por tamanho)
3. Gera gráficos automaticamente
4. Cria tabela resumo com eficiência

#### Opção 2: Análise Manual

```bash
cd build
python3 ../plot_results_simple.py
```

Ou se tiver pandas/matplotlib instalados:

```bash
pip3 install pandas matplotlib numpy
python3 ../plot_results.py
```

### Gráficos Gerados

1. **speedup_analysis.png** - Dois subgráficos:
   - **Speedup vs Tamanho**: Compara speedup teórico (Lei de Amdahl) com alcançado
   - **Eficiência**: Porcentagem do speedup teórico alcançado

2. **execution_time.png**:
   - Tempo de execução em escala log-log
   - Compara Serial CPU, GPU Global e GPU Shared Compact

### Speedup Esperado (Lei de Amdahl)

Com RTX 4060 Ti (4352 CUDA cores):

| Tamanho | Fração Serial | Speedup Teórico | Speedup Alcançado (Shared) | Eficiência |
|---------|---------------|-----------------|---------------------------|------------|
| < 10 KB | 50%           | 2-5x            | Variável                  | Baixa      |
| 10-100 KB | 20%         | 5-10x           | 2-5x                      | Moderada   |
| 100 KB - 1 MB | 10%     | 10-20x          | 4-19x                     | Alta       |
| 1-10 MB | 10%           | 10-25x          | 19-29x                    | Muito Alta |
| > 10 MB | 5%            | 15-30x          | 24-25x                    | Excelente  |

## 🧪 Metodologia dos Testes

### Condições de Corrida Identificadas

1. **Acesso simultâneo à STT** (State Transition Table)
   - **Solução**: STT como read-only, sem necessidade de sincronização

2. **Escrita de resultados (matches encontrados)**
   - **Solução**: Uso de `atomicAdd()` para contadores globais

3. **Compartilhamento de memória entre threads**
   - **Solução**: Cada thread processa porção independente do texto

### Compactação da STT

A **State Transition Table** é compactada usando 3 vetores:

- **VI (Vetor de Índices)**: Índice inicial no VE para cada estado
- **VE (Vetor de Entrada)**: Caracteres de entrada válidos
- **VS (Vetor de Saída)**: Estados de destino

**Redução de memória:** ~75% do tamanho original

### Ambiente de Testes Recomendado

```
Hardware:
- GPU: NVIDIA RTX 4060 Ti (8GB VRAM)
- CPU: [Especificar seu processador]
- RAM: [Especificar quantidade]
- Storage: SSD recomendado

Software:
- SO: Ubuntu 22.04 LTS / Windows 11
- CUDA: 12.x
- Driver NVIDIA: 545.xx ou superior
```

## 📝 Para o Relatório do Trabalho

### Itens Obrigatórios (10 pontos)

#### 1. Compreensão do Artigo (2.5 pts)

**Problema identificado:**
- IDS como Snort usa algoritmo AC para comparar pacotes
- Consome 70-80% do tempo de CPU
- Gargalo em redes de alta velocidade

**Abordagem paralela do artigo:**
- Paralelização de dados usando CUDA
- Teste de diferentes memórias GPU (global, textura, compartilhada)
- Compactação da STT para caber na memória compartilhada

#### 2. Proposta de Abordagem Paralela (2.5 pts)

**Condições de corrida:**
- Acesso simultâneo à STT (resolvido: read-only)
- Contadores de matches (resolvido: atomic operations)
- Buffers de saída (resolvido: buffers independentes)

**Tratamento:**
- STT marcada como `const` na GPU
- `atomicAdd()` para agregação de resultados
- Cada thread processa intervalo independente

#### 3. Metodologia de Testes (2.5 pts)

**Tamanhos de instâncias:** 1 KB, 10 KB, 100 KB, 1 MB, 10 MB, 50 MB, 100 MB, 500 MB, 1 GB

**Instâncias utilizadas:**
- Dados sintéticos (texto em inglês replicado)
- Padrões do Snort (assinaturas de IDS)

**Especificação do ambiente:**
[Preencher com suas especificações]

#### 4. Qualidade da Apresentação (2.5 pts)

- Código bem documentado
- README completo
- Resultados em CSV
- Gráficos de comparação (gerar com Python/Excel)

## 📊 Exemplo de Resultados

### Tabela Resumo (Amostra)

```
====================================================================================================
RESUMO DOS RESULTADOS (MÉDIAS DE 5 ITERAÇÕES)
====================================================================================================

📊 Tamanho: 1024 KB
Método                         Tempo (ms)      Speedup      Eficiência      Throughput (Mcps)
----------------------------------------------------------------------------------------------------
Serial_CPU                     7.86            1.00         5.0%            133.43
GPU_Global                     2.64            2.97         14.9%           415.03
GPU_Shared_Compact             0.42            18.66        93.7%           2672.91

📊 Tamanho: 10240 KB
Método                         Tempo (ms)      Speedup      Eficiência      Throughput (Mcps)
----------------------------------------------------------------------------------------------------
Serial_CPU                     76.64           1.00         5.0%            136.82
GPU_Global                     96.97           0.79         4.0%            108.13
GPU_Shared_Compact             2.67            28.71        100.0%          3927.89

📊 Tamanho: 1048576 KB (1 GB)
Método                         Tempo (ms)      Speedup      Eficiência      Throughput (Mcps)
----------------------------------------------------------------------------------------------------
Serial_CPU                     7523.11         1.00         5.0%            142.73
GPU_Global                     529.72          14.20        71.3%           2027.00
GPU_Shared_Compact             296.86          25.34        100.0%          3617.01
```

### Interpretação dos Resultados

✅ **GPU Shared Compact** alcança:
- **24-29x speedup** em datasets grandes (>10 MB)
- **93-100% de eficiência** comparado ao teórico (Lei de Amdahl)
- **3.6 Gcps** de throughput em datasets de 1 GB
- Desempenho superior ao artigo original (19x speedup reportado)

⚠️ **GPU Global** mostra:
- Bom desempenho em datasets muito grandes (>512 MB)
- Overhead de latência penaliza datasets pequenos
- 14x speedup em 1 GB (inferior ao Shared Compact)

📉 **Overhead da GPU** é significativo para datasets < 100 KB:
- Tempo de transferência de dados domina
- Serial CPU pode ser mais rápido nestes casos

## 🎯 Próximos Passos

1. ✅ Compilar o projeto
2. ✅ Executar testes rápidos
3. ✅ Executar experimentos completos
4. ✅ Gerar gráficos dos resultados
5. ⬜ Escrever relatório
6. ⬜ Preparar apresentação

## 📚 Referências

- Aho, A.; Corasick, M. (1975). "Efficient string matching"
- Silva Júnior et al. (2017). "Compactação do Algoritmo de Comparação de Strings do Snort" - WSCAD 2017
- NVIDIA CUDA Programming Guide
- Snort IDS Documentation

## 👥 Autor(es)

[Seu nome e dos membros do grupo]

## 📄 Licença

Este projeto é desenvolvido para fins acadêmicos na disciplina TN741 - Computação de Alto Desempenho da UFRRJ.

---

**Última atualização:** Novembro 2025