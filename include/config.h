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

// Configurações CUDA - seguindo o artigo original
// Global: blockSize=100 (como autor)
// Shared: blockSize=1000 (como autor)
#define BLOCK_SIZE_GLOBAL 128   // Bloco pequeno para global
#define BLOCK_SIZE_SHARED 1024  // Bloco máximo para shared
#define GRID_SIZE 512

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
    int* VI;           // Vetor de Indices (offset no VE/VS para cada estado)
    int* NE;           // Numero de Entradas por estado (para busca O(1) do fim)
    unsigned char* VE; // Vetor de Entrada (caracteres)
    int* VS;           // Vetor de Saida (proximo estado)
    int* est0;         // Tabela de lookup direto para estado 0 (256 entradas)
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