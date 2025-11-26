#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

long linhas = 200000;  //Quantidade MÁXIMA de linhas da STT
long qtdChaves=5000; // Quantidade de chaves para a máquina AC
char contents[5000][1000]; // Chaves da máquina AC (com tamanhos máximos de 1000 caracteres)

void preencherContents() //Transfere as chaves do arquivo (HD) apara o vetor contents (RAM)
{
	char url[]="contents.txt"; //Arquivo que contem as chaves da máquina AC	
	char ch;
	long nContents=0;
	long cont=0;
	FILE *arq;
	
	arq = fopen(url, "r");
	if(arq == NULL)
	    printf("Erro, nao foi possivel abrir o arquivo\n");
	else
	    while( (ch=fgetc(arq))!= EOF )
		{
			if(ch == '\n') {nContents++;cont=0;}
			else {
			contents[nContents][cont] = ch;
			cont++; }
		}
			
	fclose(arq);
}

int main( void ) {
	long lin,col,estado,i,contEstado,verificaPrimeiraInsercao,maxEstado,refEstZero,fluxoNormal;
	long contPrincipal=0;
// ALOCAÇÃO DINAMICA DA MATRIZ
	int **STT; 
	int w,m=linhas,n=256; 
	if(n<1||m<1) 
	{ printf("Erro! Parametro Invalido"); 
	} 
	STT=(int **) calloc(m,sizeof(int*)); 
	if (STT==NULL) { 
	printf("Erro! Memoria Insuficiente");  
	} 
	for (w=0;w<m;w++) 
	{ 
	STT[w]=(int *)calloc(n,sizeof(int)); 
	if(STT[w]==NULL) { 
	printf("Erro! Memoria Insuficiente");  
	} }

//PREENCHIMENTO INICIAL DA STT COM -1 EM TODAS AS POSIÇÕES
	for (lin = 0 ; lin < linhas ; lin++)
	for (col = 0 ; col < 256 ; col++)
		STT[lin][col] = -1;
	
	preencherContents();
	
	verificaPrimeiraInsercao=0;
	contEstado=1;
	maxEstado=0;

	for(i=0;i<qtdChaves;i++) //PERCORRE TODAS AS CHAVES PARA CONSTRUIR A STT
	{
	estado=0;	//RETORNA PARA A LINHA 0 DA STT
	verificaPrimeiraInsercao=0; //ZERA O FLAG DE PRIMEIRA INSERÇÃO
	refEstZero=0;
	fluxoNormal=1;
	for (lin = 0 ; contents[i][lin] != '\0' ; lin++) //PERCORRE A CHAVE ATUAL ATÉ O ÚLTIMO CARACTERE
	{		
		if(STT[estado][contents[i][lin]] == -1 && verificaPrimeiraInsercao == 0) //SE NÃO HOUVER O CARACTERE NA LINHA ATUAL DA STT, INSERE-O
		{
			verificaPrimeiraInsercao=1;
			STT[estado][contents[i][lin]] = contEstado;
			refEstZero=contEstado;
			fluxoNormal=0;
			estado++;
			contEstado++;
		}
		else if(STT[estado][contents[i][lin]] != -1 && fluxoNormal==1 && verificaPrimeiraInsercao!=1) {estado=STT[estado][contents[i][lin]];} //SE JÁ HOUVER O CARACTERE NA LINHA ATUAL DA STT E NÃO HOUVE INSERÇÃO ANTERIORMENTE, SETA O APONTADOR DO ESTADO ATUAL PARA O ESTADO QUE A STT INDICAR
		else if(verificaPrimeiraInsercao == 1){
		if(fluxoNormal == 0) //O FLUXO NÃO SERÁ NORMAL QUANDO ESTE VIER DO ESTADO 0
		{
		estado=refEstZero;
		STT[estado][contents[i][lin]] = contEstado;
		estado++;
		contEstado++;
		fluxoNormal=1;
		}
		else
		{
		STT[estado][contents[i][lin]] = contEstado;
			estado++;
			contEstado++;
		}
		
		}
	}
	}
	
	//resultado
	for (lin = 0 ; lin < linhas; lin++){ printf("Estado %d qtd = %d\n",lin-1,contPrincipal); contPrincipal=0;
	for (col = 0 ; col < 256 ; col++){
		if(STT[lin][col] != -1)
		{printf("[%d][%c]%d ",lin,col,STT[lin][col]);contPrincipal++;}}}
		
	//escrever resultado final no arquivo resultado.txt
	char url3[]="resultado.txt";
	char ch3;
	FILE *arq3;
	
	arq3 = fopen(url3, "a");
	if(arq3 == NULL)
		printf("Erro, nao foi possivel abrir o arquivo\n");
	else{
		for (lin = 0 ; lin < linhas; lin++){ fprintf(arq3, "Estado %d qtd %d\n", lin-1, contPrincipal); contPrincipal=0;
		for (col = 0 ; col < 256 ; col++){
			if(STT[lin][col] != -1)
			{contPrincipal++;}}}
		fclose(arq3);
	}
	return 0;
}