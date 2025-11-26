
set terminal png size 1200,800 enhanced font 'Arial,12'
set output 'results/execution_time.png'

set xlabel "Tamanho da Entrada (KB)" font 'Arial,14'
set ylabel "Tempo de Execução (ms)" font 'Arial,14'
set title "Tempo de Execução por Tamanho de Entrada" font 'Arial,16'
set logscale xy
set grid
set key top left

plot 'results/plot_data.dat' using 1:2 with linespoints lw 2 pt 5 ps 1.5 lc rgb 'blue' title 'Serial CPU', \
     '' using 1:4 with linespoints lw 2 pt 6 ps 1.5 lc rgb 'red' title 'GPU Global', \
     '' using 1:6 with linespoints lw 2 pt 4 ps 1.5 lc rgb 'green' title 'GPU Shared Compact'
