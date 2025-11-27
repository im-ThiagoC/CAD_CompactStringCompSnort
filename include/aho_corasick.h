#ifndef AHO_CORASICK_H
#define AHO_CORASICK_H

#include "config.h"
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

// Estrutura para armazenar lista de outputs
typedef struct {
    int* data;
    int size;
    int capacity;
} OutputList;

// Estrutura de um nó do autômato
typedef struct {
    int goto_table[MAX_ALPHABET_SIZE];  // Tabela goto
    int failure;                         // Link de falha
    OutputList output;                   // Padrões que terminam neste estado
} ACNode;

// Estrutura do Autômato Aho-Corasick
typedef struct {
    ACNode* states;
    char** patterns;
    int num_patterns;
    int num_states;
    int states_capacity;
} AhoCorasick;

// Criar/destruir autômato
AhoCorasick* createAhoCorasick();
void destroyAhoCorasick(AhoCorasick* ac);

// Adicionar padrões
void addPattern(AhoCorasick* ac, const char* pattern);
void addPatterns(AhoCorasick* ac, char** patterns, int num_patterns);

// Construir autômato
void buildAhoCorasick(AhoCorasick* ac);

// Busca serial
long long searchSerial(AhoCorasick* ac, const char* text, size_t text_size);

// Exportar STT (State Transition Table)
void exportSTT(AhoCorasick* ac, int** stt, int** failure_table);

// Exportar STT compactada
CompactedSTT exportCompactedSTT(AhoCorasick* ac);

// Exportar outputs para GPU
int* exportOutputCounts(AhoCorasick* ac);

// Getters
int getNumStates(AhoCorasick* ac);
int getNumPatterns(AhoCorasick* ac);

// Debug
void printAutomaton(AhoCorasick* ac);

// Funções auxiliares para OutputList
void initOutputList(OutputList* list);
void addOutput(OutputList* list, int pattern_id);
void freeOutputList(OutputList* list);

// Funções GPU
PerformanceMetrics searchGPU_Global(const char* text, size_t text_size, 
                                     int* d_stt, int* d_failure,
                                     int* h_output_counts, 
                                     int num_states);

PerformanceMetrics searchGPU_SharedCompact(const char* text, size_t text_size,
                                            CompactedSTT compact_stt,
                                            int* failure_table,
                                            int* output_counts);

#ifdef __cplusplus
}
#endif

#endif // AHO_CORASICK_H