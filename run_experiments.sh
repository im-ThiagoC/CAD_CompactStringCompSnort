#!/bin/bash
# =============================================================================
# Script para executar experimentos de escalabilidade do Aho-Corasick GPU
# 
# Autor: Thiago Carvalho
# Data: 27/11/2025
# Curso: TN741 - Computação de Alto Desempenho - UFRRJ
#
# Uso: ./run_experiments.sh
# =============================================================================

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Executando Experimentos de Escalabilidade (5 iterações)  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

cd "$(dirname "$0")/build"

# Verificar se o executável existe
if [ ! -f "./aho_corasick" ]; then
    echo "❌ Erro: Executável não encontrado. Compile primeiro com 'make'."
    exit 1
fi

# Limpar resultados anteriores
echo "🧹 Limpando resultados anteriores..."
rm -f ../results/experiment_*.csv

# Executar experimento de escalabilidade (opção 4)
echo "🚀 Executando experimento de escalabilidade..."
echo "⏱️  Isso pode levar alguns minutos (5 iterações por tamanho)..."
echo ""

echo "4" | ./aho_corasick

# Verificar se os CSVs foram gerados
csv_count=$(ls -1 ../results/experiment_*.csv 2>/dev/null | wc -l)

if [ $csv_count -eq 0 ]; then
    echo ""
    echo "❌ Erro: Nenhum arquivo CSV foi gerado."
    exit 1
fi

echo ""
echo "✅ Experimentos concluídos! Gerados $csv_count arquivos CSV."
echo ""

# Executar script Python para gerar gráficos
echo "📊 Gerando gráficos..."
cd ..

if command -v python3 &> /dev/null; then
    python3 plot_results.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✅ Gráficos gerados com sucesso!"
        echo ""
        echo "📁 Arquivos gerados:"
        echo "   - results/speedup_analysis.png"
        echo "   - results/throughput_analysis.png"
        echo "   - results/execution_time.png"
        echo "   - results/summary_results.csv"
    else
        echo ""
        echo "⚠️  Erro ao gerar gráficos. Verifique se as dependências Python estão instaladas:"
        echo "   pip3 install pandas matplotlib numpy"
    fi
else
    echo "⚠️  Python3 não encontrado. Instale Python para gerar os gráficos."
    echo "   Dados CSV disponíveis em: results/experiment_*.csv"
fi

echo ""
echo "🎉 Processo completo!"
