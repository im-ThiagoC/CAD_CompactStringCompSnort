#ifndef CONFIG_H
#define CONFIG_H

#ifdef __cplusplus
extern "C" {
#endif

// Configurações gerais
#define MAX_ALPHABET_SIZE 256
#define MAX_PATTERN_LENGTH 100
#define MAX_PATTERNS 1000
#define MAX_STATES 50000

// Configurações CUDA
#define BLOCK_SIZE 256
#define GRID_SIZE 1000

// Tipos de memória GPU
typedef enum {
    GLOBAL_MEMORY,
    TEXTURE_MEMORY,
    SHARED_MEMORY_COMPACT
} MemoryType;

// Estrutura para resultados
typedef struct {
    float execution_time_ms;
    float kernel_time_ms;
    float throughput_mcps;  // Mega caracteres por segundo
    long long matches_found;
} PerformanceMetrics;

// Estrutura da STT compactada
typedef struct {
    int* VI;           // Vetor de Índices
    unsigned char* VE; // Vetor de Entrada
    int* VS;           // Vetor de Saída
    int num_states;
    int total_entries;
} CompactedSTT;

// Estrutura para outputs na GPU
typedef struct {
    int* outputs;      // Array de pattern IDs (linearizado)
    int* output_offsets; // Offset para cada estado
    int* output_counts;  // Quantidade de outputs por estado
    int total_outputs;
} GPUOutputs;

#ifdef __cplusplus
}
#endif

#endif // CONFIG_H