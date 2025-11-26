#!/usr/bin/env python3
"""
Gera gráficos de desempenho do algoritmo Aho-Corasick
Versão simplificada usando apenas bibliotecas padrão
Para versão completa com pandas/matplotlib, use plot_results.py
"""

import csv
import glob
import os
import json

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
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['Size_KB'] = size_kb
                row['Time(ms)'] = float(row['Time(ms)'])
                row['Speedup'] = float(row['Speedup'])
                row['Throughput(Mcps)'] = float(row['Throughput(Mcps)'])
                row['Matches'] = int(row['Matches'])
                all_data.append(row)
    
    return all_data

def calculate_theoretical_speedup(size_kb, num_cores=4352):
    """
    Calcula speedup teórico esperado baseado no modelo de Amdahl
    Para GPU: Speedup = 1 / (S + (1-S)/P)
    """
    if size_kb < 10:
        serial_fraction = 0.5
    elif size_kb < 100:
        serial_fraction = 0.2
    elif size_kb < 1000:
        serial_fraction = 0.1
    else:
        serial_fraction = 0.05
    
    parallel_fraction = 1 - serial_fraction
    theoretical_speedup = 1 / (serial_fraction + (parallel_fraction / num_cores))
    
    return min(theoretical_speedup, 100)

def generate_summary_table(data):
    """Gera tabela resumo em formato texto e CSV"""
    
    # Organizar dados por tamanho
    by_size = {}
    for row in data:
        size = row['Size_KB']
        if size not in by_size:
            by_size[size] = []
        by_size[size].append(row)
    
    # Criar arquivo CSV resumo
    summary_file = 'results/summary_results.csv'
    with open(summary_file, 'w', newline='') as f:
        fieldnames = ['Size_KB', 'Method', 'Time(ms)', 'Speedup', 'Throughput(Mcps)', 
                      'Theoretical_Speedup', 'Efficiency(%)', 'Matches']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        print("\n" + "="*100)
        print("RESUMO DOS RESULTADOS (MÉDIAS)")
        print("="*100)
        
        for size in sorted(by_size.keys()):
            print(f"\n📊 Tamanho: {size} KB")
            print(f"{'Método':<30} {'Tempo (ms)':<15} {'Speedup':<12} {'Eficiência':<15} {'Throughput (Mcps)'}")
            print("-" * 100)
            
            theoretical = calculate_theoretical_speedup(size)
            
            for row in by_size[size]:
                efficiency = (row['Speedup'] / theoretical * 100) if theoretical > 0 else 0
                efficiency = min(efficiency, 100)
                
                print(f"{row['Method']:<30} {row['Time(ms)']:<15.2f} "
                      f"{row['Speedup']:<12.2f} {efficiency:<15.1f}% "
                      f"{row['Throughput(Mcps)']:.2f}")
                
                # Escrever no CSV
                writer.writerow({
                    'Size_KB': size,
                    'Method': row['Method'],
                    'Time(ms)': f"{row['Time(ms)']:.2f}",
                    'Speedup': f"{row['Speedup']:.2f}",
                    'Throughput(Mcps)': f"{row['Throughput(Mcps)']:.2f}",
                    'Theoretical_Speedup': f"{theoretical:.2f}",
                    'Efficiency(%)': f"{efficiency:.1f}",
                    'Matches': row['Matches']
                })
    
    print(f"\n✅ Tabela resumo salva: {summary_file}")
    return summary_file

def generate_gnuplot_scripts(data):
    """Gera scripts Gnuplot para criar gráficos"""
    
    # Organizar dados por método
    by_method = {}
    for row in data:
        method = row['Method']
        if method not in by_method:
            by_method[method] = []
        by_method[method].append(row)
    
    # Criar arquivo de dados para Gnuplot
    data_file = 'results/plot_data.dat'
    with open(data_file, 'w') as f:
        f.write("# Size_KB\tSerial_Time\tSerial_Speedup\tGlobal_Time\tGlobal_Speedup\t"
                "Shared_Time\tShared_Speedup\tTheoretical_Speedup\n")
        
        sizes = sorted(set(row['Size_KB'] for row in data))
        for size in sizes:
            size_data = {row['Method']: row for row in data if row['Size_KB'] == size}
            
            serial = size_data.get('Serial_CPU', {})
            global_gpu = size_data.get('GPU_Global', {})
            shared = size_data.get('GPU_Shared_Compact', {})
            theoretical = calculate_theoretical_speedup(size)
            
            f.write(f"{size}\t"
                   f"{serial.get('Time(ms)', 0)}\t{serial.get('Speedup', 1)}\t"
                   f"{global_gpu.get('Time(ms)', 0)}\t{global_gpu.get('Speedup', 0)}\t"
                   f"{shared.get('Time(ms)', 0)}\t{shared.get('Speedup', 0)}\t"
                   f"{theoretical}\n")
    
    # Script Gnuplot para speedup
    speedup_script = 'results/plot_speedup.gnu'
    with open(speedup_script, 'w') as f:
        f.write("""
set terminal png size 1600,800 enhanced font 'Arial,12'
set output 'results/speedup_analysis.png'

set multiplot layout 1,2 title "Análise de Speedup - Aho-Corasick GPU"

# Gráfico 1: Speedup vs Tamanho
set xlabel "Tamanho da Entrada (KB)" font 'Arial,14'
set ylabel "Speedup vs CPU Serial" font 'Arial,14'
set title "Speedup: Teórico vs Alcançado" font 'Arial,16'
set logscale xy
set grid
set key bottom right

plot 'results/plot_data.dat' using 1:8 with linespoints lw 2 pt 7 ps 1.5 lc rgb 'gray' title 'Teórico (Lei de Amdahl)', \\
     '' using 1:5 with linespoints lw 2 pt 6 ps 1.5 lc rgb 'red' title 'GPU Global', \\
     '' using 1:7 with linespoints lw 2 pt 4 ps 1.5 lc rgb 'green' title 'GPU Shared Compact'

# Gráfico 2: Eficiência
set xlabel "Tamanho da Entrada (KB)" font 'Arial,14'
set ylabel "Eficiência (% do Teórico)" font 'Arial,14'
set title "Eficiência da Implementação GPU" font 'Arial,16'
set logscale x
unset logscale y
set yrange [0:105]
set grid

plot 'results/plot_data.dat' using 1:(($5/$8)*100 > 100 ? 100 : ($5/$8)*100) with linespoints lw 2 pt 6 ps 1.5 lc rgb 'red' title 'GPU Global', \\
     '' using 1:(($7/$8)*100 > 100 ? 100 : ($7/$8)*100) with linespoints lw 2 pt 4 ps 1.5 lc rgb 'green' title 'GPU Shared Compact', \\
     100 with lines lw 1 lc rgb 'green' dt 2 notitle

unset multiplot
""")
    
    # Script Gnuplot para tempo de execução
    time_script = 'results/plot_time.gnu'
    with open(time_script, 'w') as f:
        f.write("""
set terminal png size 1200,800 enhanced font 'Arial,12'
set output 'results/execution_time.png'

set xlabel "Tamanho da Entrada (KB)" font 'Arial,14'
set ylabel "Tempo de Execução (ms)" font 'Arial,14'
set title "Tempo de Execução por Tamanho de Entrada" font 'Arial,16'
set logscale xy
set grid
set key top left

plot 'results/plot_data.dat' using 1:2 with linespoints lw 2 pt 5 ps 1.5 lc rgb 'blue' title 'Serial CPU', \\
     '' using 1:4 with linespoints lw 2 pt 6 ps 1.5 lc rgb 'red' title 'GPU Global', \\
     '' using 1:6 with linespoints lw 2 pt 4 ps 1.5 lc rgb 'green' title 'GPU Shared Compact'
""")
    
    print(f"\n✅ Scripts Gnuplot gerados:")
    print(f"   - {speedup_script}")
    print(f"   - {time_script}")
    print(f"\n💡 Para gerar os gráficos, execute:")
    print(f"   gnuplot {speedup_script}")
    print(f"   gnuplot {time_script}")
    
    # Tentar executar gnuplot se disponível
    try:
        import subprocess
        result = subprocess.run(['gnuplot', '--version'], 
                               capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\n📊 Gerando gráficos com Gnuplot...")
            subprocess.run(['gnuplot', speedup_script], check=True)
            subprocess.run(['gnuplot', time_script], check=True)
            print("✅ Gráficos gerados com sucesso!")
            print(f"   - results/speedup_analysis.png")
            print(f"   - results/execution_time.png")
        else:
            print("\n⚠️  Gnuplot não encontrado. Instale com: sudo apt install gnuplot")
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
        print("\n⚠️  Gnuplot não encontrado. Instale com: sudo apt install gnuplot")

def main():
    print("🔍 Analisando resultados dos experimentos...")
    
    # Carregar dados
    data = load_all_results()
    
    if data is None:
        print("❌ Erro: Não foi possível carregar os dados")
        return
    
    num_sizes = len(set(row['Size_KB'] for row in data))
    print(f"✅ Carregados {len(data)} registros de {num_sizes} tamanhos diferentes")
    
    # Gerar tabela resumo
    print("\n📋 Gerando tabela resumo...")
    generate_summary_table(data)
    
    # Gerar scripts Gnuplot
    print("\n📊 Gerando scripts de visualização...")
    generate_gnuplot_scripts(data)
    
    print("\n✅ Análise completa!")
    print("📁 Resultados salvos em: results/")

if __name__ == "__main__":
    main()
