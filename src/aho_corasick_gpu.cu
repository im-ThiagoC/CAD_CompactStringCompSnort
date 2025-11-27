#include "../include/aho_corasick.h"
#include "../include/utils.h"
#include <cuda_runtime.h>
#include <stdio.h>

// =============================================================================
// IMPLEMENTACAO GPU DO ALGORITMO AHO-CORASICK COM STT COMPACTADA
// Baseado no artigo: "Compactacao do Algoritmo de Comparacao de Strings do Snort"
// 
// A STT compactada usa tres vetores:
// - VI (Vetor de Indices): aponta para o inicio das entradas de cada estado
// - VE (Vetor de Entradas): armazena os caracteres de entrada
// - VS (Vetor de Estados): armazena o proximo estado para cada entrada
//
// Otimizacao: Estado 0 usa lookup direto (est0) para acesso O(1)
// =============================================================================

// Limites para shared memory (RTX 4060 Ti tem 48KB por bloco)
#define SH_NUM_STATES 1024   // Maximo de estados
#define SH_NUM_ENTRIES 4096  // Maximo de entradas no VE/VS
#define MAX_PATTERN_LEN 32   // Overlap para padroes que cruzam bordas

// ========== KERNEL 1: Memoria Global com STT Compactada ==========
// Este kernel usa a STT compactada armazenada em memoria GLOBAL
// Para comparacao justa com o kernel de shared memory (como no artigo original)
__global__ void acSearchGlobalCompact(const char* text, size_t text_size,
                                       const int* VI, const int* NE,
                                       const unsigned char* VE, const int* VS,
                                       const int* failure, const int* output_counts,
                                       const int* est0,
                                       int num_states, int total_entries,
                                       long long* matches_out) {
    
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long total_threads = gridDim.x * blockDim.x;
    
    // Divisao do texto entre threads com overlap
    long chars_per_thread = (text_size + total_threads - 1) / total_threads;
    long start_pos = idx * chars_per_thread;
    long end_pos = min(start_pos + chars_per_thread, text_size);
    
    // Adicionar overlap para capturar padroes nas bordas
    // Cada thread processa um pouco antes do seu inicio (exceto a primeira)
    long overlap_start = (idx == 0) ? 0 : max(0L, start_pos - MAX_PATTERN_LEN);
    
    if (start_pos >= text_size) return;
    
    long local_matches = 0;
    int state = 0;
    
    // Processar desde overlap_start para construir estado correto
    for (long i = overlap_start; i < end_pos; i++) {
        unsigned char ch = (unsigned char)(text[i]);
        
        int next_state = -1;
        
        // FUNCAO GOTO: usando STT compactada da memoria global
        if (state == 0) {
            // Estado 0: lookup direto O(1)
            next_state = est0[ch];
        } else if (state < num_states) {
            // Outros estados: busca linear
            int vi_start = VI[state];
            if (vi_start >= 0) {
                int num_ent = NE[state];
                int vi_end = vi_start + num_ent;
                if (vi_end > total_entries) vi_end = total_entries;
                
                for (int j = vi_start; j < vi_end; j++) {
                    if (VE[j] == ch) {
                        next_state = VS[j];
                        break;
                    }
                }
            }
        }
        
        // FUNCAO FAILURE
        while (next_state == -1 && state != 0) {
            state = failure[state];
            
            if (state == 0) {
                next_state = est0[ch];
            } else if (state < num_states) {
                int vi_start = VI[state];
                if (vi_start >= 0) {
                    int num_ent = NE[state];
                    int vi_end = vi_start + num_ent;
                    if (vi_end > total_entries) vi_end = total_entries;
                    
                    for (int j = vi_start; j < vi_end; j++) {
                        if (VE[j] == ch) {
                            next_state = VS[j];
                            break;
                        }
                    }
                }
            }
        }
        
        // Atualiza estado
        if (next_state >= 0) {
            state = next_state;
        }
        
        // FUNCAO OUTPUT - so conta matches na regiao propria (nao no overlap)
        if (i >= start_pos && state < num_states && output_counts[state] > 0) {
            local_matches += output_counts[state];
        }
    }
    
    atomicAdd((unsigned long long*)matches_out, (unsigned long long)local_matches);
}

// ========== KERNEL 2: Memoria Compartilhada com STT Compactada ==========
// Este kernel usa a STT compactada carregada em shared memory
// Seguindo a abordagem do autor do artigo (BLOCK_SIZE maior = melhor ocupancia)
__global__ void acSearchSharedCompact(const char* text, size_t text_size,
                                       const int* VI, const int* NE,
                                       const unsigned char* VE, const int* VS,
                                       const int* failure, const int* output_counts,
                                       const int* est0,
                                       int num_states, int total_entries,
                                       long long* matches_out) {
    
    // Alocar memoria compartilhada para os vetores do automato
    __shared__ int s_VI[SH_NUM_STATES];
    __shared__ int s_NE[SH_NUM_STATES];
    __shared__ int s_failure[SH_NUM_STATES];
    __shared__ int s_output[SH_NUM_STATES];
    __shared__ unsigned char s_VE[SH_NUM_ENTRIES];
    __shared__ int s_VS[SH_NUM_ENTRIES];
    __shared__ int s_est0[256];
    
    int tid = threadIdx.x;
    int block_size = blockDim.x;
    
    int states_to_load = min(num_states, SH_NUM_STATES);
    int entries_to_load = min(total_entries, SH_NUM_ENTRIES);
    
    // Carregamento cooperativo - mais eficiente para GPUs modernas
    // Cada thread carrega elementos diferentes
    for (int i = tid; i < states_to_load; i += block_size) {
        s_VI[i] = VI[i];
        s_NE[i] = NE[i];
        s_failure[i] = failure[i];
        s_output[i] = output_counts[i];
    }
    
    for (int i = tid; i < entries_to_load; i += block_size) {
        s_VE[i] = VE[i];
        s_VS[i] = VS[i];
    }
    
    for (int i = tid; i < 256; i += block_size) {
        s_est0[i] = est0[i];
    }
    
    __syncthreads();
    
    // Calcular faixa de texto
    long idx = blockIdx.x * blockDim.x + threadIdx.x;
    long total_threads = gridDim.x * blockDim.x;
    
    long chars_per_thread = (text_size + total_threads - 1) / total_threads;
    long start_pos = idx * chars_per_thread;
    long end_pos = min(start_pos + chars_per_thread, text_size);
    
    // Adicionar overlap para capturar padroes nas bordas
    long overlap_start = (idx == 0) ? 0 : max(0L, start_pos - MAX_PATTERN_LEN);
    
    if (start_pos >= text_size) return;
    
    long local_matches = 0;
    int state = 0;
    
    // Processar desde overlap_start para construir estado correto
    for (long i = overlap_start; i < end_pos; i++) {
        unsigned char ch = (unsigned char)(text[i]);
        
        int next_state = -1;
        
        // FUNCAO GOTO: usando shared memory
        if (state == 0) {
            next_state = s_est0[ch];
        } else if (state < states_to_load) {
            int vi_start = s_VI[state];
            if (vi_start >= 0) {
                int num_ent = s_NE[state];
                int vi_end = vi_start + num_ent;
                if (vi_end > entries_to_load) vi_end = entries_to_load;
                
                for (int j = vi_start; j < vi_end; j++) {
                    if (s_VE[j] == ch) {
                        next_state = s_VS[j];
                        break;
                    }
                }
            }
        }
        
        // FUNCAO FAILURE
        while (next_state == -1 && state != 0) {
            state = s_failure[state];
            
            if (state == 0) {
                next_state = s_est0[ch];
            } else if (state < states_to_load) {
                int vi_start = s_VI[state];
                if (vi_start >= 0) {
                    int num_ent = s_NE[state];
                    int vi_end = vi_start + num_ent;
                    if (vi_end > entries_to_load) vi_end = entries_to_load;
                    
                    for (int j = vi_start; j < vi_end; j++) {
                        if (s_VE[j] == ch) {
                            next_state = s_VS[j];
                            break;
                        }
                    }
                }
            }
        }
        
        if (next_state >= 0) {
            state = next_state;
        }
        
        // FUNCAO OUTPUT - so conta matches na regiao propria (nao no overlap)
        if (i >= start_pos && state < states_to_load && s_output[state] > 0) {
            local_matches += s_output[state];
        }
    }
    
    atomicAdd((unsigned long long*)matches_out, (unsigned long long)local_matches);
}

// ========== FUNCOES HOST ==========

PerformanceMetrics searchGPU_Global(const char* text, size_t text_size,
                                     int* h_stt, int* h_failure,
                                     int* h_output_counts,
                                     int num_states) {
    // Esta funcao agora usa a STT COMPACTADA em memoria global
    // para comparacao justa com a versao de shared memory
    PerformanceMetrics metrics;
    metrics.execution_time_ms = 0;
    metrics.kernel_time_ms = 0;
    metrics.throughput_mcps = 0;
    metrics.matches_found = 0;
    
    // Criar STT compactada a partir da STT completa
    int* h_VI = (int*)malloc(num_states * sizeof(int));
    int* h_NE = (int*)calloc(num_states, sizeof(int));
    int* h_est0 = (int*)malloc(256 * sizeof(int));
    
    // Contar entradas
    int total_entries = 0;
    for (int i = 0; i < num_states; i++) {
        for (int j = 0; j < MAX_ALPHABET_SIZE; j++) {
            if (h_stt[i * MAX_ALPHABET_SIZE + j] != -1) {
                total_entries++;
            }
        }
    }
    
    unsigned char* h_VE = (unsigned char*)malloc(total_entries * sizeof(unsigned char));
    int* h_VS = (int*)malloc(total_entries * sizeof(int));
    
    // Preencher vetores compactados
    int entry_idx = 0;
    for (int i = 0; i < num_states; i++) {
        int start_idx = entry_idx;
        int num_ent = 0;
        
        for (int j = 0; j < MAX_ALPHABET_SIZE; j++) {
            int next = h_stt[i * MAX_ALPHABET_SIZE + j];
            if (next != -1) {
                h_VE[entry_idx] = (unsigned char)j;
                h_VS[entry_idx] = next;
                entry_idx++;
                num_ent++;
            }
        }
        
        h_VI[i] = (num_ent > 0) ? start_idx : -1;
        h_NE[i] = num_ent;
    }
    
    // Preencher est0
    for (int j = 0; j < 256; j++) {
        h_est0[j] = h_stt[j];  // Estado 0
    }
    
    cudaEvent_t start, stop, kernel_start, kernel_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    
    // Alocar memoria GPU
    char* d_text;
    int* d_VI;
    int* d_NE;
    unsigned char* d_VE;
    int* d_VS;
    int* d_failure;
    int* d_output_counts;
    int* d_est0;
    long long* d_matches;
    
    CUDA_CHECK(cudaMalloc(&d_text, text_size));
    CUDA_CHECK(cudaMalloc(&d_VI, num_states * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_NE, num_states * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_VE, total_entries * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_VS, total_entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_failure, num_states * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output_counts, num_states * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_est0, 256 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_matches, sizeof(long long)));
    
    cudaEventRecord(start);
    
    CUDA_CHECK(cudaMemcpy(d_text, text, text_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_VI, h_VI, num_states * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_NE, h_NE, num_states * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_VE, h_VE, total_entries * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_VS, h_VS, total_entries * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_failure, h_failure, num_states * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_counts, h_output_counts, num_states * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_est0, h_est0, 256 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_matches, 0, sizeof(long long)));
    
    int gridSize = GRID_SIZE;
    int blockSize = BLOCK_SIZE_GLOBAL;  // Menor bloco para global
    
    cudaEventRecord(kernel_start);
    acSearchGlobalCompact<<<gridSize, blockSize>>>(d_text, text_size, 
                                                    d_VI, d_NE, d_VE, d_VS,
                                                    d_failure, d_output_counts,
                                                    d_est0,
                                                    num_states, total_entries,
                                                    d_matches);
    cudaEventRecord(kernel_stop);
    
    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess) {
        printf("ERRO no kernel GlobalCompact: %s\n", cudaGetErrorString(kernel_error));
    }
    
    long long h_matches = 0;
    CUDA_CHECK(cudaMemcpy(&h_matches, d_matches, sizeof(long long), cudaMemcpyDeviceToHost));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_time, kernel_time;
    cudaEventElapsedTime(&total_time, start, stop);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);
    
    metrics.execution_time_ms = total_time;
    metrics.kernel_time_ms = kernel_time;
    metrics.throughput_mcps = calculateThroughput(text_size, kernel_time);
    metrics.matches_found = h_matches;
    
    cudaFree(d_text);
    cudaFree(d_VI);
    cudaFree(d_NE);
    cudaFree(d_VE);
    cudaFree(d_VS);
    cudaFree(d_failure);
    cudaFree(d_output_counts);
    cudaFree(d_est0);
    cudaFree(d_matches);
    
    free(h_VI);
    free(h_NE);
    free(h_VE);
    free(h_VS);
    free(h_est0);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    
    return metrics;
}

PerformanceMetrics searchGPU_SharedCompact(const char* text, size_t text_size,
                                            CompactedSTT h_compact,
                                            int* h_failure,
                                            int* h_output_counts) {
    PerformanceMetrics metrics;
    metrics.execution_time_ms = 0;
    metrics.kernel_time_ms = 0;
    metrics.throughput_mcps = 0;
    metrics.matches_found = 0;
    
    if (h_compact.num_states > SH_NUM_STATES) {
        printf("AVISO: Estados (%d) > limite shared memory (%d)\n", 
               h_compact.num_states, SH_NUM_STATES);
    }
    
    if (h_compact.total_entries > SH_NUM_ENTRIES) {
        printf("AVISO: Entradas (%d) > limite shared memory (%d)\n",
               h_compact.total_entries, SH_NUM_ENTRIES);
    }
    
    cudaEvent_t start, stop, kernel_start, kernel_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    
    char* d_text;
    int* d_VI;
    int* d_NE;
    unsigned char* d_VE;
    int* d_VS;
    int* d_failure;
    int* d_output_counts;
    int* d_est0;
    long long* d_matches;
    
    CUDA_CHECK(cudaMalloc(&d_text, text_size));
    CUDA_CHECK(cudaMalloc(&d_VI, h_compact.num_states * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_NE, h_compact.num_states * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_VE, h_compact.total_entries * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_VS, h_compact.total_entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_failure, h_compact.num_states * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output_counts, h_compact.num_states * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_est0, 256 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_matches, sizeof(long long)));
    
    cudaEventRecord(start);
    
    CUDA_CHECK(cudaMemcpy(d_text, text, text_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_VI, h_compact.VI, h_compact.num_states * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_NE, h_compact.NE, h_compact.num_states * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_VE, h_compact.VE, h_compact.total_entries * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_VS, h_compact.VS, h_compact.total_entries * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_failure, h_failure, h_compact.num_states * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_counts, h_output_counts, h_compact.num_states * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_est0, h_compact.est0, 256 * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_matches, 0, sizeof(long long)));
    
    int gridSize = GRID_SIZE;
    int blockSize = BLOCK_SIZE_SHARED;  // Maior bloco para shared (melhor ocupância)
    
    cudaEventRecord(kernel_start);
    acSearchSharedCompact<<<gridSize, blockSize>>>(d_text, text_size, 
                                                    d_VI, d_NE, d_VE, d_VS,
                                                    d_failure, d_output_counts,
                                                    d_est0,
                                                    h_compact.num_states,
                                                    h_compact.total_entries,
                                                    d_matches);
    cudaEventRecord(kernel_stop);
    
    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess) {
        printf("ERRO no kernel SharedCompact: %s\n", cudaGetErrorString(kernel_error));
    }
    
    long long h_matches = 0;
    CUDA_CHECK(cudaMemcpy(&h_matches, d_matches, sizeof(long long), cudaMemcpyDeviceToHost));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float total_time, kernel_time;
    cudaEventElapsedTime(&total_time, start, stop);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);
    
    metrics.execution_time_ms = total_time;
    metrics.kernel_time_ms = kernel_time;
    metrics.throughput_mcps = calculateThroughput(text_size, kernel_time);
    metrics.matches_found = h_matches;
    
    cudaFree(d_text);
    cudaFree(d_VI);
    cudaFree(d_NE);
    cudaFree(d_VE);
    cudaFree(d_VS);
    cudaFree(d_failure);
    cudaFree(d_output_counts);
    cudaFree(d_est0);
    cudaFree(d_matches);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    
    return metrics;
}
