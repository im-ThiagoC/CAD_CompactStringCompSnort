#!/usr/bin/env python3
"""
Gera gráficos de desempenho do algoritmo Aho-Corasick GPU

Compara versões de memória global vs compartilhada.

Autor: Thiago Carvalho
Data: 27/11/2025
Curso: TN741 - Computação de Alto Desempenho - UFRRJ
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def load_all_results(results_dir='results'):
    """Carrega todos os arquivos CSV de resultados"""
    csv_files = sorted(glob.glob(f'{results_dir}/experiment_*kb.csv'))
    
    if not csv_files:
        print(f"⚠️  Nenhum arquivo CSV encontrado em {results_dir}/")
        return None
    
    all_data = []
    
    for csv_file in csv_files:
        # Extrair tamanho do arquivo
        filename = os.path.basename(csv_file)
        size_kb = int(filename.replace('experiment_', '').replace('kb.csv', ''))
        
        # Ler CSV
        df = pd.read_csv(csv_file)
        df['Size_KB'] = size_kb
        all_data.append(df)
    
    # Combinar todos os dados
    combined_df = pd.concat(all_data, ignore_index=True)
    return combined_df

def calculate_theoretical_speedup(size_kb, num_cores=4352):
    """
    Calcula speedup teórico esperado baseado no modelo de Amdahl
    Para GPU: Speedup = 1 / (S + (1-S)/P)
    Onde S = fração serial (overhead), P = núcleos paralelos
    """
    # Overhead diminui com tamanho do problema
    if size_kb < 10:
        serial_fraction = 0.5  # 50% overhead para datasets pequenos
    elif size_kb < 100:
        serial_fraction = 0.2  # 20% overhead
    elif size_kb < 1000:
        serial_fraction = 0.1  # 10% overhead
    else:
        serial_fraction = 0.05  # 5% overhead para grandes datasets
    
    # Lei de Amdahl
    parallel_fraction = 1 - serial_fraction
    theoretical_speedup = 1 / (serial_fraction + (parallel_fraction / num_cores))
    
    # Limitar por fatores práticos (memória bandwidth, etc)
    max_practical_speedup = min(theoretical_speedup, 100)
    
    return max_practical_speedup

def plot_speedup_comparison(df):
    """Gera gráfico comparando speedup teórico vs alcançado"""
    
    # Preparar dados por método
    methods = df['Method'].unique()
    methods = [m for m in methods if m != 'Serial_CPU']
    
    sizes = sorted(df['Size_KB'].unique())
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico 1: Speedup vs Tamanho
    colors = {'GPU_Global': '#e74c3c', 'GPU_Shared_Compact': '#2ecc71'}
    markers = {'GPU_Global': 'o', 'GPU_Shared_Compact': 's'}
    
    for method in methods:
        method_data = df[df['Method'] == method].sort_values('Size_KB')
        ax1.plot(method_data['Size_KB'], method_data['Speedup'], 
                marker=markers.get(method, 'o'), 
                label=method.replace('_', ' '),
                linewidth=2, markersize=8,
                color=colors.get(method, 'blue'))
    
    # Adicionar speedup teórico
    theoretical_speedups = [calculate_theoretical_speedup(s) for s in sizes]
    ax1.plot(sizes, theoretical_speedups, 
            '--', label='Teórico (Lei de Amdahl)', 
            linewidth=2, color='gray', alpha=0.7)
    
    ax1.set_xlabel('Tamanho da Entrada (KB)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Speedup vs CPU Serial', fontsize=12, fontweight='bold')
    ax1.set_title('Speedup: Teórico vs Alcançado', fontsize=14, fontweight='bold')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3, which='both')
    ax1.legend(fontsize=10, loc='lower right')
    
    # Linha de referência
    ax1.axhline(y=1, color='black', linestyle=':', alpha=0.5, linewidth=1)
    
    # Gráfico 2: Eficiência (% do teórico alcançado)
    for method in methods:
        method_data = df[df['Method'] == method].sort_values('Size_KB')
        
        efficiencies = []
        for _, row in method_data.iterrows():
            theoretical = calculate_theoretical_speedup(row['Size_KB'])
            efficiency = (row['Speedup'] / theoretical) * 100
            efficiencies.append(min(efficiency, 100))  # Cap at 100%
        
        ax2.plot(method_data['Size_KB'], efficiencies,
                marker=markers.get(method, 'o'),
                label=method.replace('_', ' '),
                linewidth=2, markersize=8,
                color=colors.get(method, 'blue'))
    
    ax2.set_xlabel('Tamanho da Entrada (KB)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Eficiência (% do Teórico)', fontsize=12, fontweight='bold')
    ax2.set_title('Eficiência da Implementação GPU', fontsize=14, fontweight='bold')
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 105)
    
    # Linha de referência 100%
    ax2.axhline(y=100, color='green', linestyle='--', alpha=0.5, linewidth=1, label='100% Eficiência')
    
    plt.tight_layout()
    plt.savefig('results/speedup_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico salvo: results/speedup_analysis.png")
    plt.close()

def plot_throughput(df):
    """Gera gráfico de throughput"""
    methods = df['Method'].unique()
    methods = [m for m in methods if m != 'Serial_CPU']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'GPU_Global': '#e74c3c', 'GPU_Shared_Compact': '#2ecc71', 'Serial_CPU': '#3498db'}
    markers = {'GPU_Global': 'o', 'GPU_Shared_Compact': 's', 'Serial_CPU': '^'}
    
    # Plotar todos os métodos incluindo Serial
    for method in df['Method'].unique():
        method_data = df[df['Method'] == method].sort_values('Size_KB')
        ax.plot(method_data['Size_KB'], method_data['Throughput(Mcps)'],
               marker=markers.get(method, 'o'),
               label=method.replace('_', ' '),
               linewidth=2, markersize=8,
               color=colors.get(method, 'blue'))
    
    ax.set_xlabel('Tamanho da Entrada (KB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Throughput (Mcps - Milhões chars/seg)', fontsize=12, fontweight='bold')
    ax.set_title('Throughput de Processamento', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/throughput_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico salvo: results/throughput_analysis.png")
    plt.close()

def plot_execution_time(df):
    """Gera gráfico de tempo de execução"""
    methods = df['Method'].unique()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'GPU_Global': '#e74c3c', 'GPU_Shared_Compact': '#2ecc71', 'Serial_CPU': '#3498db'}
    markers = {'GPU_Global': 'o', 'GPU_Shared_Compact': 's', 'Serial_CPU': '^'}
    
    for method in methods:
        method_data = df[df['Method'] == method].sort_values('Size_KB')
        ax.plot(method_data['Size_KB'], method_data['Time(ms)'],
               marker=markers.get(method, 'o'),
               label=method.replace('_', ' '),
               linewidth=2, markersize=8,
               color=colors.get(method, 'blue'))
    
    ax.set_xlabel('Tamanho da Entrada (KB)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tempo de Execução (ms)', fontsize=12, fontweight='bold')
    ax.set_title('Tempo de Execução por Tamanho de Entrada', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3, which='both')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig('results/execution_time.png', dpi=300, bbox_inches='tight')
    print("✅ Gráfico salvo: results/execution_time.png")
    plt.close()

def generate_summary_table(df):
    """Gera tabela resumo em CSV"""
    summary = df.groupby(['Size_KB', 'Method']).agg({
        'Time(ms)': 'mean',
        'Speedup': 'mean',
        'Throughput(Mcps)': 'mean',
        'Matches': 'first'
    }).reset_index()
    
    summary_file = 'results/summary_results.csv'
    summary.to_csv(summary_file, index=False, float_format='%.2f')
    print(f"✅ Tabela resumo salva: {summary_file}")
    
    # Imprimir resumo na tela
    print("\n" + "="*80)
    print("RESUMO DOS RESULTADOS (MÉDIAS)")
    print("="*80)
    
    for size in sorted(summary['Size_KB'].unique()):
        print(f"\n📊 Tamanho: {size} KB")
        size_data = summary[summary['Size_KB'] == size]
        print(f"{'Método':<30} {'Tempo (ms)':<15} {'Speedup':<12} {'Throughput (Mcps)'}")
        print("-" * 80)
        for _, row in size_data.iterrows():
            print(f"{row['Method']:<30} {row['Time(ms)']:<15.2f} {row['Speedup']:<12.2f} {row['Throughput(Mcps)']:.2f}")

def main():
    print("🔍 Analisando resultados dos experimentos...")
    
    # Carregar dados
    df = load_all_results()
    
    if df is None:
        print("❌ Erro: Não foi possível carregar os dados")
        return
    
    print(f"✅ Carregados {len(df)} registros de {len(df['Size_KB'].unique())} tamanhos diferentes")
    
    # Gerar gráficos
    print("\n📊 Gerando gráficos...")
    plot_speedup_comparison(df)
    plot_throughput(df)
    plot_execution_time(df)
    
    # Gerar tabela resumo
    print("\n📋 Gerando tabela resumo...")
    generate_summary_table(df)
    
    print("\n✅ Análise completa!")
    print("📁 Resultados salvos em: results/")

if __name__ == "__main__":
    main()
