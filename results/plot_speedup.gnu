
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

plot 'results/plot_data.dat' using 1:8 with linespoints lw 2 pt 7 ps 1.5 lc rgb 'gray' title 'Teórico (Lei de Amdahl)', \
     '' using 1:5 with linespoints lw 2 pt 6 ps 1.5 lc rgb 'red' title 'GPU Global', \
     '' using 1:7 with linespoints lw 2 pt 4 ps 1.5 lc rgb 'green' title 'GPU Shared Compact'

# Gráfico 2: Eficiência
set xlabel "Tamanho da Entrada (KB)" font 'Arial,14'
set ylabel "Eficiência (% do Teórico)" font 'Arial,14'
set title "Eficiência da Implementação GPU" font 'Arial,16'
set logscale x
unset logscale y
set yrange [0:105]
set grid

plot 'results/plot_data.dat' using 1:(($5/$8)*100 > 100 ? 100 : ($5/$8)*100) with linespoints lw 2 pt 6 ps 1.5 lc rgb 'red' title 'GPU Global', \
     '' using 1:(($7/$8)*100 > 100 ? 100 : ($7/$8)*100) with linespoints lw 2 pt 4 ps 1.5 lc rgb 'green' title 'GPU Shared Compact', \
     100 with lines lw 1 lc rgb 'green' dt 2 notitle

unset multiplot
