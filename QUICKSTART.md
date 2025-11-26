# 🚀 Guia Rápido - Análise com Gráficos

## Setup Inicial (Uma vez apenas)

### 1. Instalar Gnuplot

```bash
sudo apt install -y gnuplot
```

### 2. Compilar o Projeto

```bash
cd ~/Code/CAD_CompactStringCompSnort
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

## Executar Análise Completa

### Opção 1: Script Automatizado (RECOMENDADO)

```bash
cd ~/Code/CAD_CompactStringCompSnort/build
./run_experiments.sh
```

**Este script faz tudo:**
1. ✅ Limpa resultados anteriores
2. ✅ Executa experimento de escalabilidade (9 tamanhos: 1 KB até 1 GB)
3. ✅ Cada tamanho roda **5 iterações** e calcula média
4. ✅ Gera 2 gráficos PNG automaticamente
5. ✅ Cria tabela resumo com eficiência

**Tempo estimado:** 10-15 minutos

### Opção 2: Passo a Passo Manual

```bash
cd ~/Code/CAD_CompactStringCompSnort/build

# 1. Executar experimentos
echo "4" | ./aho_corasick

# 2. Gerar gráficos
python3 ../plot_results_simple.py
```

## Verificar Resultados

### Arquivos Gerados

```bash
ls -lh ../results/

# Você verá:
# - experiment_*kb.csv        (dados brutos de cada tamanho)
# - summary_results.csv        (tabela consolidada)
# - speedup_analysis.png       (gráfico de speedup)
# - execution_time.png         (gráfico de tempo)
```

### Ver Tabela Resumo

```bash
cat ../results/summary_results.csv
```

### Ver Gráficos

```bash
# Abrir com visualizador de imagens padrão
xdg-open ../results/speedup_analysis.png
xdg-open ../results/execution_time.png

# Ou copiar para seu diretório de trabalho
cp ../results/*.png ~/Documentos/
```

## Análise Avançada (Opcional)

### Com pandas/matplotlib

Se quiser gráficos mais sofisticados:

```bash
pip3 install pandas matplotlib numpy
python3 ../plot_results.py
```

Isso gera 3 gráficos ao invés de 2:
- `speedup_comparison.png` - Speedup teórico vs alcançado + eficiência
- `throughput_comparison.png` - Throughput (Mcps) por tamanho
- `execution_time_comparison.png` - Tempo de execução comparativo

## Interpretando os Resultados

### Speedup Analysis (speedup_analysis.png)

**Gráfico Superior: Speedup vs Tamanho**
- Linha **cinza**: Speedup teórico (Lei de Amdahl)
- Linha **vermelha**: GPU Global (memória global)
- Linha **verde**: GPU Shared Compact (memória compartilhada)

✅ **Esperado:** Linha verde próxima ou acima da cinza = Alta eficiência
⚠️ **Atenção:** Linha verde abaixo da cinza = Overhead ou gargalo

**Gráfico Inferior: Eficiência**
- Mostra % do speedup teórico alcançado
- 100% = desempenho teórico ideal
- >80% = excelente
- 50-80% = bom
- <50% = precisa otimização

### Execution Time (execution_time.png)

Compara tempo de execução:
- **Azul**: Serial CPU (baseline)
- **Vermelho**: GPU Global
- **Verde**: GPU Shared Compact

✅ **Esperado:** Verde sempre abaixo das outras linhas em datasets grandes

## Resultados Esperados

### RTX 4060 Ti (8 GB, 4352 CUDA cores)

| Tamanho | Speedup Shared | Eficiência | Throughput |
|---------|----------------|------------|------------|
| < 10 KB | 0.5x - 1x      | Baixa      | Variável   |
| 10-100 KB | 2x - 5x      | Moderada   | 300-700 Mcps |
| 100 KB - 1 MB | 5x - 19x | Alta       | 500-2700 Mcps |
| 1-10 MB | 19x - 29x      | Muito Alta | 2700-4000 Mcps |
| > 10 MB | 24x - 26x      | Excelente  | 3500-3900 Mcps |

## Troubleshooting

### "No such file or directory: experiment_*kb.csv"

**Causa:** Experimentos não foram executados ainda

**Solução:**
```bash
cd build
echo "4" | ./aho_corasick
python3 ../plot_results_simple.py
```

### "gnuplot: command not found"

**Causa:** Gnuplot não instalado

**Solução:**
```bash
sudo apt install gnuplot
```

### Gráficos não aparecem

**Causa:** Gnuplot executou mas sem interface gráfica

**Solução:** Os arquivos PNG foram criados! Verifique:
```bash
ls -lh ../results/*.png
```

### Speedup muito baixo (<5x)

**Possíveis causas:**
1. Dataset muito pequeno (overhead domina)
2. GPU não está sendo usada corretamente
3. Driver NVIDIA desatualizado

**Diagnóstico:**
```bash
# Verificar GPU
nvidia-smi

# Re-executar teste de 1 MB
cd build
./aho_corasick
# Escolha opção 1
```

Esperado para 1 MB: Speedup ~18-20x

## Para o Relatório

### Gráficos Obrigatórios

1. ✅ `speedup_analysis.png` - Mostra comparação com teórico
2. ✅ `execution_time.png` - Mostra evolução do tempo

### Tabelas Obrigatórias

1. ✅ `summary_results.csv` - Dados consolidados com eficiência

### Análise Recomendada

```bash
# Copiar arquivos para relatório
cp results/speedup_analysis.png ~/Relatorio/figuras/
cp results/execution_time.png ~/Relatorio/figuras/
cp results/summary_results.csv ~/Relatorio/dados/
```

**No relatório, incluir:**
- Gráficos com legenda explicativa
- Tabela com resultados principais (3-4 tamanhos representativos)
- Análise: Por que Shared Compact é melhor?
- Comparação com artigo original (19x vs seus resultados)
- Discussão sobre overhead em datasets pequenos

## Próximos Passos

1. ✅ Executar experimentos completos
2. ✅ Gerar gráficos
3. ⬜ Analisar resultados e identificar padrões
4. ⬜ Comparar com artigo original
5. ⬜ Escrever relatório com análise dos gráficos
6. ⬜ Preparar apresentação

---

📚 **Documentação completa:** Veja `EXPERIMENTOS.md` para detalhes técnicos
📋 **README principal:** Veja `README.md` para visão geral do projeto
