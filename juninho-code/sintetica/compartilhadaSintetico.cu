/****
**Busca vários padrões em um texto.
**Case -insensitive.
**Autor: JUNINHO
**OK
***/

#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
//#include <cudaProfiler.h>
#include <cuda_profiler_api.h>
#include <time.h>
#include <string.h>
#include "redeufs.h"

const int pkts = LINHAS; //quantidade de pacotes que será processado
const int sizeText = COLUNAS;
const long repeticoes9 = 1000000; //mudar o nome para textos
short int h_texto9[pkts*repeticoes9][sizeText];

void preencheEntrada9()
{
	int contAux = 0;

	for(int k = 0; k < repeticoes9; k++)
	{
		for(int i = 0; i < pkts; i++, contAux++)
		{
			for(int j = 0; j < sizeText; j++)
			{
				h_texto9[contAux][j] = pkt[i][j];
			}
		}
	}
}

const int h_Ponteiros[296] = {0,11,14,15,16,-1,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,-1,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,-1,91,93,94,95,96,97,98,99,100,101,102,103,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,-1,133,134,135,136,137,138,139,140,141,142,-1,143,144,-1,145,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,-1,174,175,176,177,178,179,180,181,182,183,184,185,186,-1,187,188,189,190,-1,191,192,193,194,195,196,199,200,201,202,203,204,-1,205,206,207,208,209,-1,210,211,212,213,-1,214,215,216,217,219,220,221,-1,222,223,224,225,226,227,228,229,230,231,232,-1,233,234,235,236,237,238,239,240,241,242,243,244,-1,245,246,247,248,249,250,251,252,253,-1,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,-1,272,273,274,275,276,277,278,-1,279,280,281,282,283,284,285,286,-1,287,288,289,290,291,292,293,-1,294};
const int h_Saida[294] = {208,278,137,251,1,241,82,270,287,184,228,6,2,57,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,134,83,84,85,86,87,88,89,90,91,92,93,123,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,124,125,126,127,128,129,130,131,132,133,135,136,145,179,138,139,140,141,142,143,144,146,165,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,166,167,168,169,170,171,172,173,174,175,176,177,178,180,181,182,183,185,186,187,188,189,197,190,203,191,192,193,194,195,196,198,199,200,201,202,204,205,206,207,209,210,211,212,216,213,214,215,217,218,219,220,221,222,223,224,225,226,227,229,230,231,232,233,234,235,236,237,238,239,240,242,243,244,245,246,247,248,249,250,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,271,272,273,274,275,276,277,279,280,281,282,283,284,285,286,288,289,290,291,292,293,294};
//especificamente nesse estado se não encontrar a entrada vai pro proprio estado zero. Posso mudar de hexa pra short, fazendo um programa que leia os valores do vetor e imprima em decimal.
const short int h_Entrada[294] = {0x00,0x2F,0x41,0x43,0x4C,0x50,0x55,0x56,0x63,0x70,0x80,0x00,0x4F,0x6F,0x47,0x49,0x4E,0x6F,0x00,0x67,0x00,0x69,0x00,0x6E,0x00,0x20,0x00,0x66,0x00,0x61,0x00,0x69,0x00,0x6C,0x00,0x65,0x00,0x64,0x00,0x20,0x00,0x66,0x00,0x6F,0x00,0x72,0x00,0x20,0x00,0x75,0x00,0x73,0x00,0x65,0x00,0x72,0x00,0x20,0x00,0x27,0x00,0x73,0x00,0x61,0x00,0x27,0x00,0x67,0x69,0x6E,0x20,0x66,0x61,0x69,0x6C,0x65,0x64,0x20,0x66,0x6F,0x72,0x20,0x75,0x73,0x65,0x72,0x20,0x27,0x73,0x61,0x27,0x53,0x73,0x45,0x52,0x2D,0x41,0x67,0x65,0x6E,0x74,0x3A,0x20,0x4A,0x57,0x65,0x62,0x74,0x72,0x65,0x6E,0x64,0x73,0x20,0x53,0x65,0x63,0x75,0x72,0x69,0x74,0x79,0x20,0x41,0x6E,0x61,0x6C,0x79,0x7A,0x65,0x72,0x0D,0x0A,0x61,0x76,0x61,0x31,0x2E,0x32,0x2E,0x31,0x0D,0x0A,0x45,0x52,0x41,0x6D,0x75,0x74,0x68,0x65,0x6E,0x74,0x69,0x63,0x41,0x61,0x74,0x69,0x6F,0x6E,0x20,0x75,0x6E,0x73,0x75,0x63,0x63,0x65,0x73,0x73,0x66,0x75,0x6C,0x41,0x41,0x41,0x41,0x41,0x41,0x41,0x41,0x41,0x41,0x41,0x41,0x41,0x61,0x6E,0x64,0x61,0x61,0x73,0x73,0x20,0x2D,0x63,0x69,0x73,0x73,0x73,0x40,0x69,0x73,0x73,0x6B,0x6C,0x61,0x75,0x73,0x61,0x69,0x6E,0x74,0x00,0x00,0x00,0x00,0x45,0x00,0x00,0x00,0x45,0x45,0x45,0x45,0x45,0x45,0x45,0x45,0x45,0x45,0x45,0x07,0x00,0x00,0x07,0x00,0x00,0x04,0x00,0x00,0x00,0x00,0x00,0x41,0x53,0x53,0x20,0x64,0x64,0x64,0x40,0x0A,0x69,0x6E,0x63,0x6F,0x20,0x4E,0x65,0x74,0x77,0x6F,0x72,0x6B,0x2C,0x20,0x49,0x6E,0x63,0x2E,0x45,0x52,0x53,0x49,0x4F,0x4E,0x0A,0x63,0x79,0x62,0x65,0x72,0x63,0x6F,0x70,0x79,0x62,0x65,0x72,0x63,0x6F,0x70};
const int h_est0[129] = {208,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,278,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,137,0,251,0,0,0,0,0,0,0,0,1,0,0,0,241,0,0,0,0,82,270,0,0,0,0,0,0,0,0,0,0,0,0,287,0,0,0,0,0,0,0,0,0,0,0,0,184,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,228};
//ainda vou completar o h_est0 com zeros até o maior tamanho de entrada de tabela ascii e tbm devo criar um vetor de falha pra tirar os cases e reduzir um pouco mais o tempo
const int h_falha[295] = {0, 0, 0, 0, 0, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 208, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 137, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 287, 0, 0, 0, 0, 0, 0, 137, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 137, 287, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 287, 287, 0, 0, 0, 0, 0, 0, 145, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 287, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 208, 209, 210, 211, 212, 213, 214, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 208, 209, 0, 208, 209, 0, 208, 209, 210, 211, 212, 0, 137, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 287, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 287, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 287, 288, 289, 290, 291, 292, 293, 294, 0, 0, 0, 0, 0, 287, 0, 184}; 

__device__ int gotoAC(int est_atual, int caractere) //estarei considerando estado como um simples int
{
	int est_proximo=0;
	return est_proximo; //retorna o próximo estado
}

__device__ int failAC(int est_atual, int* d_falha) //estarei considerando estado como um simples int
{
	int est_proximo=0;
	est_proximo = d_falha[est_atual];
  return est_proximo; //retorna o próximo estado
}

__device__ int outputAC(int est_atual) //estarei considerando estado como um simples int
{
if(est_atual == 5 || est_atual == 81 || est_atual == 122 || est_atual == 133 || est_atual == 136 || est_atual == 164 || est_atual == 178 || est_atual == 183 || est_atual == 196 || est_atual == 202 || est_atual == 207 || est_atual == 215 || est_atual == 227 || est_atual == 250 || est_atual == 269 || est_atual == 277 || est_atual == 286 || est_atual == 294) return 1;
else return 0;
}

__device__ long get_global_index(void) //MATRIX
{
	return blockIdx.x * blockDim.x + threadIdx.x;
} 

__global__ void ahoCorasick9(short int* d_texto9, int* d_contador9, int* d_Ponteiros, int* d_Saida, short int* d_Entrada, int* d_est0, int* d_falha) 
{ 
  long idx = get_global_index();
  
  int est_atual = 0;
  int est_proximo = 400;
  
  d_contador9[idx] = 0;
  
  __shared__ int sd_Ponteiros[296];
  __shared__ int sd_Saida[294];
  __shared__ short int sd_Entrada[294];
  __shared__ int sd_est0[129];
  // gasta 5 ms
  for(int n = 0; n < 129; n++)
  {	
	sd_Ponteiros[n] = d_Ponteiros[n];
	sd_Saida[n] = d_Saida[n];
	sd_Entrada[n] = d_Entrada[n];
	sd_est0[n] = d_est0[n];
  }
  for(int n = 64; n < 294; n++)
  {	
	sd_Ponteiros[n] = d_Ponteiros[n];
	sd_Saida[n] = d_Saida[n];
	sd_Entrada[n] = d_Entrada[n];
  }
  sd_Ponteiros[294] = d_Ponteiros[294];
  sd_Ponteiros[295] = d_Ponteiros[295];
  
  int indice,indiceAux,qtdEntradas;
  long i;
  
  for(i=(idx*sizeText); i< ((idx*sizeText) + sizeText) ; i++) //percorre o texto
  {
		//FUNÇÃO GOTO
		indice = sd_Ponteiros[est_atual];
		
		if(indice != -1)
		{			
			if(est_atual == 0 && d_texto9[i] < 0x79) {est_proximo = sd_est0[d_texto9[i]];} //seria x81
			else if (est_atual != 0){
				if(sd_Ponteiros[est_atual+1] == -1) indiceAux = sd_Ponteiros[est_atual+2]; //isso deve ta estourando o vetor... tem que fazer tratamento de erro
				else indiceAux = sd_Ponteiros[est_atual+1];
				qtdEntradas = indiceAux - indice;
				for(int n=indice; n< indiceAux; n++)
				{
					est_proximo = 400;
					if(d_texto9[i] == sd_Entrada[n]) {est_proximo = sd_Saida[n]; break;}
				}
				/* //OTIMIZAÇÃO
				est_proximo = 400;
				if(d_texto9[i] <= sd_Entrada[indice+(qtdEntradas/2)])
				for(int n=indice; n<= indice+(qtdEntradas/2); n++)
				{
					if(d_texto9[i] == sd_Entrada[n]) {est_proximo = sd_Saida[n]; break;}
				}
				else 
				for(int n=(indice+(qtdEntradas/2))+1; n< indice+qtdEntradas; n++)
				{
					if(d_texto9[i] == sd_Entrada[n]) {est_proximo = sd_Saida[n]; break;}
				} */
			}
		}
		else {est_proximo = 400;}
		
		//FIM DA FUNÇÃO GOTO
		
		if (est_proximo == 400 && est_atual != 0){ //tive que tratar pra n ficar em loop infinito 
			est_proximo = failAC(est_atual,d_falha); //tem que mudar os case daqui de est0 para 0...
			i=i-1; //para a entrada do espado de falha ser a mesma do estado atual.
		}
			
		est_atual = est_proximo;

		 if (outputAC(est_atual) != 0){
			//saida = outputAC(est_atual);
			//if( saoIguais(outputAC(est_atual),"the"))
				d_contador9[idx] = d_contador9[idx] + 1;
				
			}
  
  }
} 

void launchMyKernel(int g, int b) 
{ //CONSERTAR O CÁLCULO DA OCUPÂNCIA PARA BIDIMENSIONAL, TEM NO LAB DE MATRIZ
  int blockSize;   // The launch configurator returned block size 
  int minGridSize; // The minimum grid size needed to achieve the 
                   // maximum occupancy for a full device launch 
  int gridSize;    // The actual grid size needed, based on input size 

  cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
                                      ahoCorasick9, 0, 0); //usado caso queira calcular a ocupância máxima - nesse caso tem que comentar a linha gridSize=g;blockSize=b;
  // Round up according to array size 
  gridSize = (pkts*repeticoes9 + blockSize - 1) / blockSize; 

  gridSize=g;blockSize=b; //seta gridSize e blockSize manualmente.
  
  cudaDeviceSynchronize(); 
  printf("\nGridsize = %d e blocksise = %d", gridSize, blockSize);
  
  // calculate theoretical occupancy
  int maxActiveBlocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, 
                                                 ahoCorasick9, blockSize, 
                                                 0);

  int device;
  cudaDeviceProp props;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&props, device);

  float occupancy = (maxActiveBlocks * blockSize / props.warpSize) / 
                    (float)(props.maxThreadsPerMultiProcessor / 
                            props.warpSize); //denominador tenho o número máximo de warps e no numerador tenho o número máximo de warps ativos

  printf("\nLaunched blocks of size %d. Theoretical occupancy: %f\n", 
         blockSize, occupancy);
}

void Cleanup(void);
void RandomInit(long*, int);
void TextInit(char*);
char** alocar_matriz_real(int m, int n);

int main( void ) {
	dim3 dg(2,1,1);
	dim3 db(1,1,1);
	short int* d_texto9;
	int* d_contador9; //vetor que armazena a quantidade de ocorrências encontradas para cada padrão em sua posição correspondente.
	int h_contador9[pkts*repeticoes9];
	int gridSize;
	int blockSize;
	int* d_Ponteiros;
    int* d_Saida;
    short int* d_Entrada;
	int* d_est0;
	int* d_falha;
	cudaEvent_t start, stop;
	float time,total;
	total=0;
	
	preencheEntrada9();
	
cudaProfilerStart();
	//alocacao de memoria na GPU
	cudaMalloc( (void**)&d_texto9, pkts*repeticoes9*sizeText*sizeof(short int) );
	cudaMalloc( (void**)&d_contador9, pkts*repeticoes9*sizeof(int) );
	cudaMalloc( (void**)&d_Ponteiros, 296*sizeof(int) );
	cudaMalloc( (void**)&d_Saida, 294*sizeof(int) );
	cudaMalloc( (void**)&d_Entrada, 294*sizeof(short int) );
	cudaMalloc( (void**)&d_est0, 129*sizeof(int) );
	cudaMalloc( (void**)&d_falha, 295*sizeof(int) );
	
	// transfere o array h_texto para a memória da GPU
	cudaMemcpy( d_texto9, h_texto9, pkts*repeticoes9*sizeText*sizeof(short int),cudaMemcpyHostToDevice );
	cudaMemcpy( d_Ponteiros, h_Ponteiros, 296*sizeof(int),cudaMemcpyHostToDevice );
	cudaMemcpy( d_Saida, h_Saida, 294*sizeof(int),cudaMemcpyHostToDevice );
	cudaMemcpy( d_Entrada, h_Entrada, 294*sizeof(short int),cudaMemcpyHostToDevice );
	cudaMemcpy( d_est0, h_est0, 129*sizeof(int),cudaMemcpyHostToDevice );
	cudaMemcpy( d_falha, h_falha, 295*sizeof(int),cudaMemcpyHostToDevice );
	
	//Cálculo da ocupância
	blockSize=1000;gridSize=pkts*repeticoes9/1000;
	launchMyKernel(gridSize, blockSize);
	
	total=0;
	for(int k=0; k < 50; k++)
	{
	cudaEventCreate(&start); 
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	//chamada ao kernel
	ahoCorasick9<<< gridSize, blockSize >>>(d_texto9, d_contador9, d_Ponteiros, d_Saida, d_Entrada, d_est0, d_falha); 
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time,start,stop); //Computes the elapsed time between two events (in milliseconds with a resolution of around 0.5 microseconds).
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	total=total+time;
	}

	// transfere resultado armazenado na variável d_contador para a variável da CPU
	cudaMemcpy( &h_contador9, d_contador9, pkts*repeticoes9*sizeof(int),cudaMemcpyDeviceToHost );
	
		printf("O tempo para a GPU fazer o processamento foi de %f milisegundos\n", total/50);

	//depuracao
//	for(int i=0;i < pkts*repeticoes; i++)
//	{
//		printf("h_contador[%d]-> %d\n",i,h_contador[i]);
//	}
	
	int tot=0;
	//Soma a quantidade de padrões encontrados
	for(int i=0;i < pkts*repeticoes9; i++)
	{
		tot = tot + h_contador9[i];
	}
	
	printf("TOTAL: %d padroes encontrados no texto.", tot);
	
	Cleanup();
cudaProfilerStop();
	return 0;
}

void Cleanup(void)
{
//	if(d_a)
//		cudaFree(d_a);
}