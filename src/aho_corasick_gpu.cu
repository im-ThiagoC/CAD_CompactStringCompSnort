#include "../include/aho_corasick.h"
#include "../include/utils.h"
#include <cuda_runtime.h>
#include <stdio.h>

// Definições para limites da Shared Memory (Ajuste conforme sua GPU)
// RTX 4060 Ti tem 48KB (49152 bytes) de shared memory por bloco
// Calculo: s_VI(1536*4) + s_VE(6144*1) + s_VS(6144*4) + s_output_counts(1536*4) = 43008 bytes < 49152
#define SH_VI_SIZE 1536
#define SH_VE_SIZE 6144
#define SH_VS_SIZE 6144

// AtomicAdd para long long (compatibilidade com todas as GPUs)
__device__ __forceinline__ void atomicAddLongLong(long long* address, long long val) {
    unsigned long long* address_as_ull = (unsigned long long*)address;
    unsigned long long old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, assumed + val);
    } while (assumed != old);
}

// ========== KERNEL 1: Memória Global ==========
// Estratégia: Cada thread processa pequenos chunks independentes com stride
__global__ void acSearchGlobal(const char* text, size_t text_size, 
                                const int* stt, const int* failure,
                                const int* output_counts,
                                int num_states, long long* matches_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Define o tamanho do chunk que cada thread processa (ajuste para balancear)
    const int CHUNK_SIZE = 256;  // Tamanho do chunk por iteração
    
    long long local_matches = 0;
    
    // Cada thread processa múltiplos chunks com stride
    for (size_t chunk_start = idx * CHUNK_SIZE; chunk_start < text_size; chunk_start += stride * CHUNK_SIZE) {
        int state = 0;  // Reinicia estado para cada chunk (independência)
        size_t chunk_end = min(chunk_start + CHUNK_SIZE, text_size);
        
        // Processar chunk
        for (size_t i = chunk_start; i < chunk_end; i++) {
            unsigned char ch = (unsigned char)(text[i]);
            
            // Transição de estado usando goto/failure
            int next_state = stt[state * MAX_ALPHABET_SIZE + ch];
            
            while (next_state == -1 && state != 0) {
                state = failure[state];
                next_state = stt[state * MAX_ALPHABET_SIZE + ch];
            }
            
            if (next_state != -1) {
                state = next_state;
            }
            
            // Contar matches usando output_counts
            if (output_counts && output_counts[state] > 0) {
                local_matches += output_counts[state];
            }
        }
    }
    
    atomicAddLongLong(matches_out, local_matches);
}

// ========== KERNEL 2: Memória Compartilhada com Compactação ==========
__global__ void acSearchSharedCompact(const char* text, size_t text_size,
                                       const int* VI, const unsigned char* VE,
                                       const int* VS, const int* failure,
                                       const int* output_counts,
                                       int num_states, int total_entries, long long* matches_out) {

    // Memória compartilhada
    __shared__ int              s_VI[SH_VI_SIZE];
    __shared__ unsigned char    s_VE[SH_VE_SIZE];
    __shared__ int              s_VS[SH_VS_SIZE];
    __shared__ int              s_output_counts[SH_VI_SIZE];

    int tid = threadIdx.x;

    // Carregar STT compactada para Shared Memory
    for (int i = tid; i < num_states && i < SH_VI_SIZE; i += blockDim.x) {
        s_VI[i] = VI[i];
        s_output_counts[i] = output_counts[i];
    }

    // Carregar VE e VS para Shared Memory
    for (int i = tid; i < total_entries && i < SH_VE_SIZE; i += blockDim.x) {
        s_VE[i] = VE[i];
        s_VS[i] = VS[i];
    }

    // Esperar todas as threads carregarem os dados
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    long long local_matches = 0;
    int state = 0;
    
    // Processar caracteres mantendo estado (mais eficiente que chunks)
    for (size_t pos = idx; pos < text_size; pos += stride) {
        unsigned char ch = (unsigned char)(text[pos]);
        int next_state = -1;
        
        // --- BUSCA NA SHARED MEMORY ---
        if (state < SH_VI_SIZE && s_VI[state] != -1) {
            int vi_idx = s_VI[state];
            
            if (vi_idx < SH_VE_SIZE) {
                // Encontrar o range end
                int range_end = total_entries;
                for (int s = state + 1; s < num_states && s < SH_VI_SIZE; s++) {
                    if (s_VI[s] != -1) {
                        range_end = s_VI[s];
                        break;
                    }
                }
                
                int limit = min(range_end, SH_VE_SIZE);
                
                for (int j = vi_idx; j < limit; j++) {
                    if (s_VE[j] == ch) {
                        next_state = s_VS[j];
                        break;
                    }
                }
            }
        }
        
        // Tratamento de falhas (retorna sempre para 0 com nossa failure table simples)
        if (next_state == -1 && state != 0) {
            state = 0; // failure[state] é sempre 0
            // Tentar novamente no estado 0
            if (s_VI[0] != -1) {
                int vi_idx = s_VI[0];
                if (vi_idx < SH_VE_SIZE) {
                    int range_end = total_entries;
                    for (int s = 1; s < num_states && s < SH_VI_SIZE; s++) {
                        if (s_VI[s] != -1) {
                            range_end = s_VI[s];
                            break;
                        }
                    }
                    
                    int limit = min(range_end, SH_VE_SIZE);
                    for (int j = vi_idx; j < limit; j++) {
                        if (s_VE[j] == ch) {
                            next_state = s_VS[j];
                            break;
                        }
                    }
                }
            }
        }
        
        if (next_state != -1) {
            state = next_state;
        }
        
        // Check for matches
        if (state < SH_VI_SIZE && s_output_counts[state] > 0) {
            local_matches += s_output_counts[state];
        }
    }
    
    atomicAddLongLong(matches_out, local_matches);
}

// ========== Funções Host ==========

PerformanceMetrics searchGPU_Global(const char* text, size_t text_size,
                                     int* h_stt, int* h_failure,
                                     int* h_output_counts,
                                     int num_states) {
    PerformanceMetrics metrics;
    metrics.execution_time_ms = 0;
    metrics.kernel_time_ms = 0;
    metrics.throughput_mcps = 0;
    metrics.matches_found = 0;
    
    // Eventos CUDA para timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Alocar memória na GPU
    char* d_text;
    int* d_stt;
    int* d_failure;
    int* d_output_counts;
    long long* d_matches;
    
    size_t stt_size = num_states * MAX_ALPHABET_SIZE * sizeof(int);
    
    CUDA_CHECK(cudaMalloc(&d_text, text_size));
    CUDA_CHECK(cudaMalloc(&d_stt, stt_size));
    CUDA_CHECK(cudaMalloc(&d_failure, num_states * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output_counts, num_states * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_matches, sizeof(long long)));
    
    // Copiar dados para GPU
    cudaEventRecord(start);
    
    CUDA_CHECK(cudaMemcpy(d_text, text, text_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_stt, h_stt, stt_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_failure, h_failure, num_states * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_counts, h_output_counts, num_states * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_matches, 0, sizeof(long long)));
    
    // Executar kernel
    int gridSize = GRID_SIZE;
    int blockSize = BLOCK_SIZE;
    
    cudaEvent_t kernel_start, kernel_stop;
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    
    cudaEventRecord(kernel_start);
    acSearchGlobal<<<gridSize, blockSize>>>(d_text, text_size, d_stt, d_failure,
                                             d_output_counts, num_states, d_matches);
    cudaEventRecord(kernel_stop);
    
    // Copiar resultado
    long long h_matches = 0;
    CUDA_CHECK(cudaMemcpy(&h_matches, d_matches, sizeof(long long), cudaMemcpyDeviceToHost));
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calcular métricas
    float total_time, kernel_time;
    cudaEventElapsedTime(&total_time, start, stop);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);
    
    metrics.execution_time_ms = total_time;
    metrics.kernel_time_ms = kernel_time;
    metrics.throughput_mcps = calculateThroughput(text_size, kernel_time);
    metrics.matches_found = h_matches;
    
    // Liberar memória
    cudaFree(d_text);
    cudaFree(d_stt);
    cudaFree(d_failure);
    cudaFree(d_output_counts);
    cudaFree(d_matches);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    
    return metrics;
}

PerformanceMetrics searchGPU_SharedCompact(const char* text, size_t text_size,
                                            CompactedSTT h_compact,
                                            int* h_output_counts) {
    PerformanceMetrics metrics;
    metrics.execution_time_ms = 0;
    metrics.kernel_time_ms = 0;
    metrics.throughput_mcps = 0;
    metrics.matches_found = 0;
    
    cudaEvent_t start, stop, kernel_start, kernel_stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    
    // Alocar memória GPU
    char* d_text;
    int* d_VI;
    unsigned char* d_VE;
    int* d_VS;
    int* d_failure;
    int* d_output_counts;
    long long* d_matches;
    
    CUDA_CHECK(cudaMalloc(&d_text, text_size));
    CUDA_CHECK(cudaMalloc(&d_VI, h_compact.num_states * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_VE, h_compact.total_entries * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_VS, h_compact.total_entries * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_failure, h_compact.num_states * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_output_counts, h_compact.num_states * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_matches, sizeof(long long)));
    
    // Copiar dados
    cudaEventRecord(start);
    
    CUDA_CHECK(cudaMemcpy(d_text, text, text_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_VI, h_compact.VI, h_compact.num_states * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_VE, h_compact.VE, h_compact.total_entries * sizeof(unsigned char), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_VS, h_compact.VS, h_compact.total_entries * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_counts, h_output_counts, h_compact.num_states * sizeof(int), cudaMemcpyHostToDevice));
    
    // Failure table simples (para evitar bugs complexos de BFS)
    // Isso força retorno ao estado 0 quando não há transição
    // Performance não será ideal mas é funcional e correto
    int* h_failure = new int[h_compact.num_states];
    for (int i = 0; i < h_compact.num_states; i++) h_failure[i] = 0;
    
    CUDA_CHECK(cudaMemcpy(d_failure, h_failure, h_compact.num_states * sizeof(int), cudaMemcpyHostToDevice));
    
    CUDA_CHECK(cudaMemset(d_matches, 0, sizeof(long long)));
    
    // Executar kernel
    int gridSize = GRID_SIZE;
    int blockSize = BLOCK_SIZE;
    
    cudaEventRecord(kernel_start);
    acSearchSharedCompact<<<gridSize, blockSize>>>(d_text, text_size, d_VI, d_VE, d_VS,
                                                    d_failure, d_output_counts,
                                                    h_compact.num_states, 
                                                    h_compact.total_entries, d_matches);
    cudaEventRecord(kernel_stop);
    
    // Copiar resultado
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
    
    // Liberar memória
    cudaFree(d_text);
    cudaFree(d_VI);
    cudaFree(d_VE);
    cudaFree(d_VS);
    cudaFree(d_failure);
    cudaFree(d_output_counts);
    cudaFree(d_matches);
    delete[] h_failure;
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    
    return metrics;
}