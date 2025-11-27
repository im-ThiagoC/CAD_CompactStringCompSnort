/**
 * @file utils.h
 * @brief Interface das funções utilitárias
 * 
 * Declara funções para geração de dados, verificação CUDA e métricas.
 * 
 * @author Thiago Carvalho
 * @date 27/11/2025
 * @course TN741 - Computação de Alto Desempenho - UFRRJ
 */

#ifndef UTILS_H
#define UTILS_H

#include "config.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

// Estrutura para array de padrões
typedef struct {
    char** patterns;
    int count;
} PatternArray;

// Estrutura para resultados
typedef struct {
    char method_name[64];
    PerformanceMetrics metrics;
} Result;

// Geração de dados
char* generateSyntheticText(size_t size_kb, size_t* actual_size);
PatternArray loadPatternsFromFile(const char* filename);
PatternArray generateSnortPatterns();
void freePatternArray(PatternArray* arr);

// Função auxiliar para compatibilidade
char** generateSnortPatterns_compat(int* num_patterns);

// Utilitários de arquivo
int saveResultsToCSV(const char* filename, Result* results, size_t count);

// Verificação CUDA
void checkCudaError(cudaError_t error, const char* file, int line);
#define CUDA_CHECK(err) checkCudaError(err, __FILE__, __LINE__)

void printGPUInfo();

// Cálculo de métricas
float calculateSpeedup(float serial_time, float parallel_time);
float calculateThroughput(size_t text_size, float time_ms);

#ifdef __cplusplus
}
#endif

#endif // UTILS_H