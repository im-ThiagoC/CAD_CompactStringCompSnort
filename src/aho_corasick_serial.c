/**
 * @file aho_corasick_serial.c
 * @brief Implementação serial do algoritmo Aho-Corasick
 * 
 * Contém a construção do autômato (goto, failure, output) e a busca serial.
 * Também implementa a exportação da STT compactada para uso na GPU.
 * 
 * @author Thiago Carvalho
 * @date 27/11/2025
 * @course TN741 - Computação de Alto Desempenho - UFRRJ
 */

#include "../include/aho_corasick.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ===== SISTEMA DE LOGGING =====
// Controla nivel de log: 0=desligado, 1=fases principais, 2=detalhado, 3=muito detalhado
#define DEBUG_VERBOSE 0

#if DEBUG_VERBOSE >= 1
  #define LOG_PHASE(fmt, ...) printf("[PHASE] " fmt "\n", ##__VA_ARGS__)
#else
  #define LOG_PHASE(fmt, ...)
#endif

#if DEBUG_VERBOSE >= 2
  #define LOG(fmt, ...) printf("[LOG] " fmt "\n", ##__VA_ARGS__)
  #define LOG_FUNC(func) printf("[LOG] >> %s\n", func)
#else
  #define LOG(fmt, ...)
  #define LOG_FUNC(func)
#endif

#if DEBUG_VERBOSE >= 3
  #define LOG_VAR(name, val) printf("[LOG]   %s = %d\n", name, val)
  #define LOG_PTR(name, ptr) printf("[LOG]   %s = %p\n", name, (void*)ptr)
  #define LOG_STR(name, str) printf("[LOG]   %s = %s\n", name, str)
#else
  #define LOG_VAR(name, val)
  #define LOG_PTR(name, ptr)
  #define LOG_STR(name, str)
#endif
// ===== FIM DO SISTEMA DE LOGGING =====

// Funcoes auxiliares para OutputList
void initOutputList(OutputList* list) {
    LOG_FUNC("initOutputList");
    list->capacity = 4;
    list->size = 0;
    list->data = (int*)malloc(list->capacity * sizeof(int));
    LOG_PTR("list->data", list->data);
    LOG_VAR("list->capacity", list->capacity);
}

void addOutput(OutputList* list, int pattern_id) {
    LOG_FUNC("addOutput");
    LOG_VAR("pattern_id", pattern_id);
    LOG_VAR("list->size", list->size);
    LOG_VAR("list->capacity", list->capacity);
    if (list->size >= list->capacity) {
        LOG("Reallocating output list from %d to %d", list->capacity, list->capacity * 2);
        list->capacity *= 2;
        list->data = (int*)realloc(list->data, list->capacity * sizeof(int));
        LOG_PTR("list->data (after realloc)", list->data);
    }
    list->data[list->size++] = pattern_id;
    LOG_VAR("list->size (after add)", list->size);
}

void freeOutputList(OutputList* list) {
    LOG_FUNC("freeOutputList");
    LOG_PTR("list->data", list->data);
    if (list->data) {
        free(list->data);
        list->data = NULL;
    }
    list->size = 0;
    list->capacity = 0;
    LOG("OutputList freed");
}

// Criar automato
AhoCorasick* createAhoCorasick() {
    LOG_FUNC("createAhoCorasick");
    AhoCorasick* ac = (AhoCorasick*)malloc(sizeof(AhoCorasick));
    LOG_PTR("ac", ac);
    
    ac->states_capacity = 100;
    ac->states = (ACNode*)malloc(ac->states_capacity * sizeof(ACNode));
    LOG_PTR("ac->states", ac->states);
    LOG_VAR("ac->states_capacity", ac->states_capacity);
    
    ac->num_states = 1;
    ac->num_patterns = 0;
    ac->patterns = NULL;
    LOG("Initial values set: num_states=1, num_patterns=0");
    
    // Inicializar estado 0
    LOG("Initializing state 0 goto_table to -1");
    for (int i = 0; i < MAX_ALPHABET_SIZE; i++) {
        ac->states[0].goto_table[i] = -1;
    }
    ac->states[0].failure = 0;
    LOG("Initializing state 0 output list");
    initOutputList(&ac->states[0].output);
    
    LOG("createAhoCorasick completed successfully");
    return ac;
}

void destroyAhoCorasick(AhoCorasick* ac) {
    LOG_FUNC("destroyAhoCorasick");
    if (!ac) {
        LOG("ac is NULL, returning");
        return;
    }
    LOG_PTR("ac", ac);
    LOG_VAR("ac->num_states", ac->num_states);
    
    // Liberar outputs de cada estado
    LOG("Freeing output lists for %d states", ac->num_states);
    for (int i = 0; i < ac->num_states; i++) {
        freeOutputList(&ac->states[i].output);
    }
    
    // Liberar estados
    LOG("Freeing states array");
    free(ac->states);
    
    // Liberar padroes
    if (ac->patterns) {
        LOG("Freeing %d patterns", ac->num_patterns);
        for (int i = 0; i < ac->num_patterns; i++) {
            free(ac->patterns[i]);
        }
        free(ac->patterns);
    }
    
    LOG("Freeing AhoCorasick structure");
    free(ac);
    LOG("destroyAhoCorasick completed");
}

void addPattern(AhoCorasick* ac, const char* pattern) {
    LOG_FUNC("addPattern");
    LOG_STR("pattern", pattern);
    LOG_VAR("ac->num_patterns (before)", ac->num_patterns);
    
    // Realocar array de padroes se necessario
    ac->patterns = (char**)realloc(ac->patterns, (ac->num_patterns + 1) * sizeof(char*));
    LOG_PTR("ac->patterns (after realloc)", ac->patterns);
    
    ac->patterns[ac->num_patterns] = (char*)malloc(strlen(pattern) + 1);
    strcpy(ac->patterns[ac->num_patterns], pattern);
    ac->num_patterns++;
    LOG_VAR("ac->num_patterns (after)", ac->num_patterns);
}

void addPatterns(AhoCorasick* ac, char** patterns, int num_patterns) {
    LOG_FUNC("addPatterns");
    LOG_VAR("num_patterns", num_patterns);
    for (int i = 0; i < num_patterns; i++) {
        addPattern(ac, patterns[i]);
    }
    LOG("All %d patterns added", num_patterns);
}

static void buildGotoFunction(AhoCorasick* ac) {
    LOG_FUNC("buildGotoFunction");
    LOG_VAR("ac->num_patterns", ac->num_patterns);
    LOG_VAR("ac->num_states (initial)", ac->num_states);
    
    // Para cada padrao, adicionar ao automato
    for (int i = 0; i < ac->num_patterns; i++) {
        LOG("Processing pattern %d: %s", i, ac->patterns[i]);
        int current_state = 0;
        int pattern_len = strlen(ac->patterns[i]);
        LOG_VAR("pattern_len", pattern_len);
        
        for (int j = 0; j < pattern_len; j++) {
            unsigned char ch = (unsigned char)ac->patterns[i][j];
            LOG("  Pattern %d, char %d: '%c' (0x%02x), current_state=%d", i, j, (ch >= 32 && ch < 127) ? ch : '?', ch, current_state);
            
            // Se nao existe transicao, criar novo estado
            if (ac->states[current_state].goto_table[ch] == -1) {
                LOG("    Creating new state %d for char 0x%02x", ac->num_states, ch);
                ac->states[current_state].goto_table[ch] = ac->num_states;
                
                // Expandir array de estados se necessario
                if (ac->num_states >= ac->states_capacity) {
                    LOG("    Expanding states array from %d to %d", ac->states_capacity, ac->states_capacity * 2);
                    ac->states_capacity *= 2;
                    ac->states = (ACNode*)realloc(ac->states, ac->states_capacity * sizeof(ACNode));
                    LOG_PTR("    ac->states (after realloc)", ac->states);
                }
                
                // Inicializar novo estado
                LOG("    Initializing new state %d", ac->num_states);
                for (int k = 0; k < MAX_ALPHABET_SIZE; k++) {
                    ac->states[ac->num_states].goto_table[k] = -1;
                }
                ac->states[ac->num_states].failure = 0;
                initOutputList(&ac->states[ac->num_states].output);
                
                ac->num_states++;
                LOG_VAR("    ac->num_states (after increment)", ac->num_states);
            } else {
                LOG("    Transition already exists: state %d --0x%02x--> state %d", 
                    current_state, ch, ac->states[current_state].goto_table[ch]);
            }
            
            current_state = ac->states[current_state].goto_table[ch];
            LOG_VAR("    current_state (after transition)", current_state);
        }
        
        // Marcar estado final com o padrao encontrado
        LOG("  Marking state %d as output for pattern %d", current_state, i);
        addOutput(&ac->states[current_state].output, i);
    }
    
    LOG("All patterns processed, num_states=%d", ac->num_states);
    LOG("buildGotoFunction completed");
}

// Fila simples para BFS
typedef struct {
    int* data;
    int front;
    int rear;
    int capacity;
} Queue;

static Queue* createQueue(int capacity) {
    LOG_FUNC("createQueue");
    LOG_VAR("capacity", capacity);
    Queue* q = (Queue*)malloc(sizeof(Queue));
    LOG_PTR("q", q);
    q->capacity = capacity;
    q->front = 0;
    q->rear = 0;
    q->data = (int*)malloc(capacity * sizeof(int));
    LOG_PTR("q->data", q->data);
    return q;
}

static void enqueue(Queue* q, int value) {
    if (q->rear >= q->capacity) {
        q->capacity *= 2;
        q->data = (int*)realloc(q->data, q->capacity * sizeof(int));
    }
    q->data[q->rear++] = value;
}

static int dequeue(Queue* q) {
    if (q->front < q->rear) {
        return q->data[q->front++];
    }
    return -1;
}

static int isEmpty(Queue* q) {
    return q->front >= q->rear;
}

static void destroyQueue(Queue* q) {
    free(q->data);
    free(q);
}

static void buildFailureFunction(AhoCorasick* ac) {
    LOG_FUNC("buildFailureFunction");
    LOG_VAR("ac->num_states", ac->num_states);
    
    Queue* q = createQueue(ac->num_states);
    
    // Estados de profundidade 1 tem falha para 0
    LOG("Setting failure for depth 1 states");
    int depth1_count = 0;
    for (int i = 0; i < MAX_ALPHABET_SIZE; i++) {
        int state = ac->states[0].goto_table[i];
        if (state > 0) {  // CORRIGIDO: verificar > 0 (ignora -1 e 0)
            LOG("  State %d (from char 0x%02x) -> failure=0", state, i);
            ac->states[state].failure = 0;
            enqueue(q, state);
            depth1_count++;
        }
    }
    LOG("Found %d depth-1 states", depth1_count);
    
    // BFS para calcular funcao de falha
    LOG("Starting BFS to compute failure function");
    int bfs_iterations = 0;
    while (!isEmpty(q)) {
        int r = dequeue(q);
        bfs_iterations++;
        LOG("BFS iteration %d: processing state %d", bfs_iterations, r);
        
        for (int i = 0; i < MAX_ALPHABET_SIZE; i++) {
            int s = ac->states[r].goto_table[i];
            if (s == -1) continue;
            
            enqueue(q, s);
            
            // Encontrar falha
            int state = ac->states[r].failure;
            while (ac->states[state].goto_table[i] == -1 && state != 0) {
                state = ac->states[state].failure;
            }
            
            if (ac->states[state].goto_table[i] != -1) {
                ac->states[s].failure = ac->states[state].goto_table[i];
            } else {
                ac->states[s].failure = 0;
            }
            
            // Adicionar saidas do estado de falha
            OutputList* failure_output = &ac->states[ac->states[s].failure].output;
            if (failure_output->size > 0) {
                LOG("      Copying %d outputs from failure state %d to state %d", 
                    failure_output->size, ac->states[s].failure, s);
            }
            for (int j = 0; j < failure_output->size; j++) {
                addOutput(&ac->states[s].output, failure_output->data[j]);
            }
        }
    }
    
    LOG("BFS completed after %d iterations", bfs_iterations);
    destroyQueue(q);
    LOG("buildFailureFunction completed");
}

void buildAhoCorasick(AhoCorasick* ac) {
    LOG_FUNC("buildAhoCorasick");
    LOG_PTR("ac", ac);
    
    LOG_PHASE("===== PHASE 1: Building Goto Function =====");
    buildGotoFunction(ac);
    LOG_PHASE("===== PHASE 1 COMPLETED: %d states =====", ac->num_states);
    
    LOG_PHASE("===== PHASE 2: Building Failure Function =====");
    buildFailureFunction(ac);
    LOG_PHASE("===== PHASE 2 COMPLETED =====");
    
    printf("Automato construido com %d estados\n", ac->num_states);
    LOG("buildAhoCorasick completed successfully");
}

long long searchSerial(AhoCorasick* ac, const char* text, size_t text_size) {
    long long matches = 0;
    int current_state = 0;
    
    for (size_t i = 0; i < text_size; i++) {
        unsigned char ch = (unsigned char)text[i];
        
        // Seguir transição goto ou falha
        while (current_state != 0 && ac->states[current_state].goto_table[ch] == -1) {
            current_state = ac->states[current_state].failure;
        }
        
        if (ac->states[current_state].goto_table[ch] != -1) {
            current_state = ac->states[current_state].goto_table[ch];
        }
        
        // Verificar saídas
        if (ac->states[current_state].output.size > 0) {
            matches += ac->states[current_state].output.size;
        }
    }
    
    return matches;
}

void exportSTT(AhoCorasick* ac, int** stt, int** failure_table) {
    // Alocar STT
    *stt = (int*)malloc(ac->num_states * MAX_ALPHABET_SIZE * sizeof(int));
    *failure_table = (int*)malloc(ac->num_states * sizeof(int));
    
    // Preencher STT
    for (int i = 0; i < ac->num_states; i++) {
        for (int j = 0; j < MAX_ALPHABET_SIZE; j++) {
            (*stt)[i * MAX_ALPHABET_SIZE + j] = ac->states[i].goto_table[j];
        }
        (*failure_table)[i] = ac->states[i].failure;
    }
}

CompactedSTT exportCompactedSTT(AhoCorasick* ac) {
    LOG_FUNC("exportCompactedSTT");
    LOG_VAR("ac->num_states", ac->num_states);
    
    CompactedSTT compact;
    compact.num_states = ac->num_states;
    
    // Contar entradas validas
    LOG_PHASE("Exporting compacted STT for %d states", ac->num_states);
    LOG("Counting valid entries...");
    int total_entries = 0;
    for (int i = 0; i < ac->num_states; i++) {
        for (int j = 0; j < MAX_ALPHABET_SIZE; j++) {
            if (ac->states[i].goto_table[j] != -1) {
                total_entries++;
            }
        }
    }
    
    compact.total_entries = total_entries;
    LOG_VAR("total_entries", total_entries);
    
    // Alocar vetores
    LOG("Allocating compacted vectors...");
    compact.VI = (int*)malloc(ac->num_states * sizeof(int));
    LOG_PTR("compact.VI", compact.VI);
    compact.NE = (int*)calloc(ac->num_states, sizeof(int));
    LOG_PTR("compact.NE", compact.NE);
    compact.VE = (unsigned char*)malloc(total_entries * sizeof(unsigned char));
    LOG_PTR("compact.VE", compact.VE);
    compact.VS = (int*)malloc(total_entries * sizeof(int));
    LOG_PTR("compact.VS", compact.VS);
    
    // NOVO: Alocar tabela de lookup direto para estado 0 (seguindo abordagem do autor)
    compact.est0 = (int*)malloc(256 * sizeof(int));
    LOG_PTR("compact.est0", compact.est0);
    
    // Preencher tabela est0 com transicoes do estado 0
    // Isso permite acesso O(1) para o estado mais acessado
    LOG("Filling est0 lookup table for state 0...");
    for (int j = 0; j < 256; j++) {
        compact.est0[j] = ac->states[0].goto_table[j];
    }
    
    // Preencher vetores compactados
    LOG("Filling compacted vectors...");
    int entry_counter = 0;
    for (int i = 0; i < ac->num_states; i++) {
        if (i < 5) LOG("  Processing state %d", i);
        int start_index = entry_counter;
        int num_entries = 0;
        
        for (int j = 0; j < MAX_ALPHABET_SIZE; j++) {
            if (ac->states[i].goto_table[j] != -1) {
                compact.VE[entry_counter] = (unsigned char)j;
                compact.VS[entry_counter] = ac->states[i].goto_table[j];
                entry_counter++;
                num_entries++;
            }
        }
        
        compact.VI[i] = (num_entries > 0) ? start_index : -1;
        compact.NE[i] = num_entries;
        if (i < 5) LOG("    State %d: VI=%d, NE=%d", i, compact.VI[i], compact.NE[i]);
    }
    
    LOG("Compacted vectors filled successfully");
    printf("STT compactada: %d estados, %d entradas\n", ac->num_states, total_entries);
    printf("Tamanho original: %d KB\n", 
           (ac->num_states * MAX_ALPHABET_SIZE * (int)sizeof(int)) / 1024);
    printf("Tamanho compactado: %d KB\n",
           (ac->num_states * (int)sizeof(int) * 2 +  // VI + NE
            total_entries * ((int)sizeof(unsigned char) + (int)sizeof(int)) +
            256 * (int)sizeof(int)) / 1024);  // + est0
    
    return compact;
}

void printAutomaton(AhoCorasick* ac) {
    printf("\n=== Automato Aho-Corasick ===\n");
    printf("Numero de estados: %d\n", ac->num_states);
    printf("Numero de padroes: %d\n", ac->num_patterns);
    
    int max_print = ac->num_patterns < 10 ? ac->num_patterns : 10;
    for (int i = 0; i < max_print; i++) {
        printf("Padrao %d: %s\n", i, ac->patterns[i]);
    }
    if (ac->num_patterns > 10) {
        printf("... e mais %d padroes\n", ac->num_patterns - 10);
    }
}

int getNumStates(AhoCorasick* ac) {
    return ac->num_states;
}

int getNumPatterns(AhoCorasick* ac) {
    return ac->num_patterns;
}

int* exportOutputCounts(AhoCorasick* ac) {
    int* counts = (int*)calloc(ac->num_states, sizeof(int));
    
    for (int i = 0; i < ac->num_states; i++) {
        counts[i] = ac->states[i].output.size;
    }
    
    return counts;
}