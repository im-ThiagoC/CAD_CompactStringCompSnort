/**
 * @file utils.cu
 * @brief Funções utilitárias para CUDA e manipulação de dados
 * 
 * Contém funções para geração de texto sintético, carregamento de padrões,
 * verificação de erros CUDA e cálculo de métricas de desempenho.
 * 
 * @author Thiago Carvalho
 * @date 27/11/2025
 * @course TN741 - Computação de Alto Desempenho - UFRRJ
 */

#include "../include/utils.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void checkCudaError(cudaError_t error, const char* file, int line) {
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA Error at %s:%d\n", file, line);
		fprintf(stderr, "%s\n", cudaGetErrorString(error));
		exit(EXIT_FAILURE);
	}
}

void printGPUInfo() {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	
	printf("\n=== Informações da GPU ===\n");
	printf("Dispositivos CUDA encontrados: %d\n", deviceCount);
	
	for (int i = 0; i < deviceCount; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		
		printf("\nDispositivo %d: %s\n", i, prop.name);
		printf("  Compute Capability: %d.%d\n", prop.major, prop.minor);
		printf("  Multiprocessors: %d\n", prop.multiProcessorCount);
		printf("  CUDA Cores (aprox): %d\n", prop.multiProcessorCount * 128);
		printf("  Memória Global: %lu GB\n", prop.totalGlobalMem / (1024*1024*1024));
		printf("  Memória Compartilhada por Bloco: %lu KB\n", prop.sharedMemPerBlock / 1024);
		printf("  Clock: %d MHz\n", prop.clockRate / 1000);
		printf("  Warp Size: %d\n", prop.warpSize);
		printf("  Max Threads por Bloco: %d\n", prop.maxThreadsPerBlock);
	}
	printf("==========================\n\n");
}

char* generateSyntheticText(size_t size_kb, size_t* actual_size) {
	const char* base_text = 
		"The quick brown fox jumps over the lazy dog. "
		"Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
		"Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
		"Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris. ";
	
	size_t target_size = size_kb * 1024;
	size_t base_len = strlen(base_text);
	
	char* result = (char*)malloc(target_size + 1);
	if (!result) {
		if (actual_size) *actual_size = 0;
		return NULL;
	}
	
	size_t pos = 0;
	while (pos < target_size) {
		size_t copy_len = (target_size - pos < base_len) ? (target_size - pos) : base_len;
		memcpy(result + pos, base_text, copy_len);
		pos += copy_len;
	}
	
	result[target_size] = '\0';
	if (actual_size) *actual_size = target_size;
	return result;
}

PatternArray generateSnortPatterns() {
	PatternArray arr;
	arr.patterns = (char**)malloc(45 * sizeof(char*));
	arr.count = 0;
	
	if (!arr.patterns) return arr;
	
	const char* patterns[] = {
		"cmd.exe", "powershell", "/bin/sh", "/bin/bash",
		"wget", "curl", "nc.exe", "netcat",
		"union select", "1=1", "or 1=1", "' or '1'='1",
		"drop table", "exec(", "execute(",
		"<script>", "javascript:", "onerror=", "onload=",
		"alert(", "document.cookie",
		"../../../../", "..\\..\\..\\", 
		"shell_exec", "system(", "passthru(",
		"trojan", "backdoor", "rootkit", "keylogger",
		"ransomware", "cryptolocker",
		"syn flood", "ddos", "ping -f", "nmap",
		"he", "she", "his", "hers", "ushers"
	};
	
	int num_patterns = sizeof(patterns) / sizeof(patterns[0]);
	
	for (int i = 0; i < num_patterns; i++) {
		arr.patterns[i] = strdup(patterns[i]);
		if (arr.patterns[i]) arr.count++;
	}
	
	return arr;
}

// Função auxiliar para compatibilidade com main.cu
char** generateSnortPatterns_compat(int* num_patterns) {
	PatternArray arr = generateSnortPatterns();
	if (num_patterns) *num_patterns = arr.count;
	return arr.patterns;
}

PatternArray loadPatternsFromFile(const char* filename) {
	PatternArray arr = {NULL, 0};
	FILE* file = fopen(filename, "r");
	
	if (!file) {
		fprintf(stderr, "Erro ao abrir arquivo: %s\n", filename);
		fprintf(stderr, "Usando padrões padrão...\n");
		return generateSnortPatterns();
	}
	
	arr.patterns = (char**)malloc(1000 * sizeof(char*));
	if (!arr.patterns) {
		fclose(file);
		return arr;
	}
	
	char line[1024];
	while (fgets(line, sizeof(line), file)) {
		size_t len = strlen(line);
		if (len > 0 && line[len-1] == '\n') line[len-1] = '\0';
		
		if (len > 0 && line[0] != '#') {
			arr.patterns[arr.count] = strdup(line);
			if (arr.patterns[arr.count]) arr.count++;
		}
	}
	
	fclose(file);
	
	if (arr.count == 0) {
		fprintf(stderr, "Nenhum padrão encontrado no arquivo. Usando padrões padrão...\n");
		free(arr.patterns);
		return generateSnortPatterns();
	}
	
	return arr;
}

void freePatternArray(PatternArray* arr) {
	if (arr->patterns) {
		for (size_t i = 0; i < arr->count; i++) {
			free(arr->patterns[i]);
		}
		free(arr->patterns);
	}
	arr->patterns = NULL;
	arr->count = 0;
}

int saveResultsToCSV(const char* filename, Result* results, size_t count) {
	FILE* file = fopen(filename, "w");
	
	if (!file) {
		fprintf(stderr, "Erro ao criar arquivo: %s\n", filename);
		return 0;
	}
	
	fprintf(file, "Método,Tempo_Total_ms,Tempo_Kernel_ms,Throughput_Mcps,Matches_Encontrados\n");
	
	for (size_t i = 0; i < count; i++) {
		fprintf(file, "%s,%f,%f,%f,%lld\n",
				results[i].method_name,
				results[i].metrics.execution_time_ms,
				results[i].metrics.kernel_time_ms,
				results[i].metrics.throughput_mcps,
				results[i].metrics.matches_found);
	}
	
	fclose(file);
	printf("Resultados salvos em: %s\n", filename);
	return 1;
}

float calculateSpeedup(float serial_time, float parallel_time) {
	if (parallel_time <= 0) return 0;
	return serial_time / parallel_time;
}

float calculateThroughput(size_t text_size, float time_ms) {
	if (time_ms <= 0) return 0;
	return (text_size / 1000000.0f) / (time_ms / 1000.0f);
}