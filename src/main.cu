#include "../include/aho_corasick.h"
#include "../include/utils.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void runExperiment(const char* experiment_name, 
                   size_t text_size_kb) {
    printf("\n============================================================\n");
    printf("EXPERIMENTO: %s\n", experiment_name);
    printf("Tamanho do texto: %zu KB\n", text_size_kb);
    printf("============================================================\n");
    
    // 1. Preparar dados
    printf("\n[1/5] Gerando texto de entrada...\n");
    size_t text_size;
    char* text = generateSyntheticText(text_size_kb, &text_size);
    printf("✓ Texto gerado: %zu bytes\n", text_size);
    
    // 2. Carregar padrões
    printf("\n[2/5] Carregando padrões...\n");
    PatternArray pattern_array = loadPatternsFromFile("../data/patterns.txt");
    int num_patterns = pattern_array.count;
    char** patterns = pattern_array.patterns;
    printf("✓ Padrões carregados: %d\n", num_patterns);
    
    // 3. Construir autômato
    printf("\n[3/5] Construindo autômato Aho-Corasick...\n");
    AhoCorasick* ac = createAhoCorasick();
    addPatterns(ac, patterns, num_patterns);
    buildAhoCorasick(ac);
    printAutomaton(ac);
    
    // 4. Versão SERIAL (média de 5 execuções)
    printf("\n[4/5] Executando busca SERIAL (5 iterações)...\n");
    const int NUM_ITERATIONS = 5;
    float serial_times[NUM_ITERATIONS];
    long long serial_matches = 0;
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        clock_t start_serial = clock();
        serial_matches = searchSerial(ac, text, text_size);
        clock_t end_serial = clock();
        serial_times[iter] = ((float)(end_serial - start_serial) / CLOCKS_PER_SEC) * 1000.0f;
        printf("  Iteração %d: %.2f ms\n", iter + 1, serial_times[iter]);
    }
    
    // Calcular média
    float serial_time = 0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        serial_time += serial_times[i];
    }
    serial_time /= NUM_ITERATIONS;
    
    printf("✓ Busca serial concluída\n");
    printf("  Tempo médio: %.2f ms\n", serial_time);
    printf("  Matches: %lld\n", (long long)serial_matches);
    
    // 5. Versões GPU (média de 5 execuções)
    printf("\n[5/5] Executando buscas na GPU (5 iterações cada)...\n");
    
    // Exportar STT
    int* h_stt = NULL;
    int* h_failure = NULL;
    exportSTT(ac, &h_stt, &h_failure);
    
    // Exportar outputs
    int* h_output_counts = exportOutputCounts(ac);
    
    // Exportar STT compactada
    CompactedSTT compact_stt = exportCompactedSTT(ac);
    
    // GPU - Memória Global (5 iterações)
    printf("\n  [a] GPU - Memória Global...\n");
    float global_times[NUM_ITERATIONS];
    float global_kernel_times[NUM_ITERATIONS];
    float global_throughputs[NUM_ITERATIONS];
    long long global_matches = 0;
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        PerformanceMetrics m = searchGPU_Global(text, text_size, h_stt, h_failure,
                                                 h_output_counts, getNumStates(ac));
        global_times[iter] = m.execution_time_ms;
        global_kernel_times[iter] = m.kernel_time_ms;
        global_throughputs[iter] = m.throughput_mcps;
        global_matches = m.matches_found;
        printf("    Iteração %d: %.2f ms (kernel: %.2f ms)\n", iter + 1, 
               m.execution_time_ms, m.kernel_time_ms);
    }
    
    // Calcular médias Global
    float avg_global_time = 0, avg_global_kernel = 0, avg_global_throughput = 0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        avg_global_time += global_times[i];
        avg_global_kernel += global_kernel_times[i];
        avg_global_throughput += global_throughputs[i];
    }
    avg_global_time /= NUM_ITERATIONS;
    avg_global_kernel /= NUM_ITERATIONS;
    avg_global_throughput /= NUM_ITERATIONS;
    
    printf("  ✓ Tempo médio: %.2f ms (kernel: %.2f ms)\n", avg_global_time, avg_global_kernel);
    printf("    Speedup: %.2fx\n", calculateSpeedup(serial_time, avg_global_kernel));
    
    // GPU - Memória Compartilhada (5 iterações)
    printf("\n  [b] GPU - Memória Compartilhada (Compactada)...\n");
    float shared_times[NUM_ITERATIONS];
    float shared_kernel_times[NUM_ITERATIONS];
    float shared_throughputs[NUM_ITERATIONS];
    long long shared_matches = 0;
    
    for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
        PerformanceMetrics m = searchGPU_SharedCompact(text, text_size, compact_stt, h_output_counts);
        shared_times[iter] = m.execution_time_ms;
        shared_kernel_times[iter] = m.kernel_time_ms;
        shared_throughputs[iter] = m.throughput_mcps;
        shared_matches = m.matches_found;
        printf("    Iteração %d: %.2f ms (kernel: %.2f ms)\n", iter + 1,
               m.execution_time_ms, m.kernel_time_ms);
    }
    
    // Calcular médias Shared
    float avg_shared_time = 0, avg_shared_kernel = 0, avg_shared_throughput = 0;
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        avg_shared_time += shared_times[i];
        avg_shared_kernel += shared_kernel_times[i];
        avg_shared_throughput += shared_throughputs[i];
    }
    avg_shared_time /= NUM_ITERATIONS;
    avg_shared_kernel /= NUM_ITERATIONS;
    avg_shared_throughput /= NUM_ITERATIONS;
    
    printf("  ✓ Tempo médio: %.2f ms (kernel: %.2f ms)\n", avg_shared_time, avg_shared_kernel);
    printf("    Speedup: %.2fx\n", calculateSpeedup(serial_time, avg_shared_kernel));
    
    // Criar arquivo de resultados com médias
    char result_file[256];
    sprintf(result_file, "../results/experiment_%zukb.csv", text_size_kb);
    FILE* fp = fopen(result_file, "w");
    fprintf(fp, "Method,Time(ms),Speedup,Throughput(Mcps),Matches\n");
    fprintf(fp, "Serial_CPU,%.2f,1.00,%.2f,%lld\n", 
            serial_time, calculateThroughput(text_size, serial_time), (long long)serial_matches);
    fprintf(fp, "GPU_Global,%.2f,%.2f,%.2f,%lld\n",
            avg_global_kernel,
            calculateSpeedup(serial_time, avg_global_kernel),
            avg_global_throughput,
            global_matches);
    fprintf(fp, "GPU_Shared_Compact,%.2f,%.2f,%.2f,%lld\n",
            avg_shared_kernel,
            calculateSpeedup(serial_time, avg_shared_kernel),
            avg_shared_throughput,
            shared_matches);
    
    fclose(fp);
    
    // Comparar versões GPU
    if (avg_global_kernel > 0 && avg_shared_kernel > 0) {
        float improvement = ((avg_global_kernel - avg_shared_kernel) 
                            / avg_global_kernel) * 100.0f;
        printf("\n  📊 Ganho da versão Compartilhada vs Global: %.2f%%\n", improvement);
    }
    
    // Imprimir tabela resumo
    printf("\n============================================================\n");
    printf("RESUMO DOS RESULTADOS\n");
    printf("============================================================\n");
    printf("%-25s %12s %12s %12s\n", "Método", "Tempo (ms)", "Speedup", "Throughput");
    printf("------------------------------------------------------------\n");
    printf("%-25s %12.2f %11.2fx %11.2f Mcps\n", "Serial_CPU", serial_time, 1.0f,
           calculateThroughput(text_size, serial_time));
    printf("%-25s %12.2f %11.2fx %11.2f Mcps\n", "GPU_Global", 
           avg_global_kernel,
           calculateSpeedup(serial_time, avg_global_kernel),
           avg_global_throughput);
    printf("%-25s %12.2f %11.2fx %11.2f Mcps\n", "GPU_Shared_Compact",
           avg_shared_kernel,
           calculateSpeedup(serial_time, avg_shared_kernel),
           avg_shared_throughput);
    printf("============================================================\n");
    
    // Liberar memória
    free(text);
    for (int i = 0; i < num_patterns; i++) {
        free(patterns[i]);
    }
    free(patterns);
    free(h_stt);
    free(h_failure);
    free(h_output_counts);
    free(compact_stt.VI);
    free(compact_stt.VE);
    free(compact_stt.VS);
    destroyAhoCorasick(ac);
}

int main(int argc, char** argv) {
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║  Compactação do Algoritmo de Comparação de Strings (AC)   ║\n");
    printf("║  Implementação CUDA - TN741 CAD - UFRRJ                   ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    
    // Informações da GPU
    printGPUInfo();
    
    // Criar diretório de resultados
    int ret = system("mkdir -p ../results");
    (void)ret; // Suprimir warning
    
    // Menu de experimentos
    printf("\n=== MENU DE EXPERIMENTOS ===\n");
    printf("1. Teste rápido (1 MB)\n");
    printf("2. Experimento completo (múltiplos tamanhos)\n");
    printf("3. Comparação detalhada (10 MB, 50 MB, 100 MB)\n");
    printf("4. Teste de escalabilidade (1 KB até 1 GB)\n");
    
    int option;
    // Se foi passado argumento na linha de comando, usar ele
    if (argc > 1) {
        option = atoi(argv[1]);
        printf("\nOpção selecionada via argumento: %d\n", option);
    } else {
        // Caso contrário, perguntar ao usuário
        printf("\nEscolha uma opção (1-4): ");
        if (scanf("%d", &option) != 1) {
            option = 1; // Default
        }
    }
    
    switch(option) {
        case 1:
            runExperiment("Teste Rápido", 1024);  // 1 MB
            break;
            
        case 2:
            runExperiment("Experimento 1 MB", 1024);
            runExperiment("Experimento 10 MB", 10 * 1024);
            runExperiment("Experimento 50 MB", 50 * 1024);
            runExperiment("Experimento 100 MB", 100 * 1024);
            break;
            
        case 3:
            runExperiment("Comparação 10 MB", 10 * 1024);
            runExperiment("Comparação 50 MB", 50 * 1024);
            runExperiment("Comparação 100 MB", 100 * 1024);
            break;
            
        case 4: {
            size_t sizes[] = {1, 10, 100, 1024, 10*1024, 50*1024, 100*1024, 500*1024, 1024*1024};
            int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
            for (int i = 0; i < num_sizes; i++) {
                char exp_name[128];
                sprintf(exp_name, "Escalabilidade %zu KB", sizes[i]);
                runExperiment(exp_name, sizes[i]);
            }
            break;
        }
            
        default:
            printf("Opção inválida. Executando teste rápido...\n");
            runExperiment("Teste Padrão", 1024);
    }
    
    printf("\n✅ Todos os experimentos concluídos!\n");
    printf("📁 Resultados salvos em: ../results/\n");
    
    return 0;
}