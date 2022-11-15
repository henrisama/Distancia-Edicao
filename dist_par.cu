// ----------------------------------------------------------------------------
// Distância de edição paralelo
// Para compilar: nvcc dist_par.cu -o dist_par -Wall
// Para executar: ./dist_par <nome arquivo entrada>

#include <stdio.h>

int n, m, *dist, *dev_dist, *dist_res;
char *s, *r, *dev_s, *dev_r;

void aloca()
{
  cudaError_t 
    resultado_1,
    resultado_2,
    resultado_3,
    resultado_4;

  // aloca para host
  resultado_1 = cudaMallocHost((void **)&s, sizeof(char) * (n + 1));
  resultado_2 = cudaMallocHost((void **)&r, sizeof(char) * (m + 1));
  resultado_3 = cudaMallocHost((void **)&dist, sizeof(int) * (n + 1) * (m + 1));
  resultado_4 = cudaMallocHost((void **)&dist_res, sizeof(int) * (n + 1) * (m + 1));

  if (resultado_1 != cudaSuccess)
  {
    printf("%s\n", cudaGetErrorString(resultado_1));
    exit(1);
  } else if (resultado_2 != cudaSuccess)
  {
    printf("%s\n", cudaGetErrorString(resultado_2));
    exit(1);
  } else if (resultado_3 != cudaSuccess)
  {
    printf("%s\n", cudaGetErrorString(resultado_3));
    exit(1);
  } else if (resultado_4 != cudaSuccess)
  {
    printf("%s\n", cudaGetErrorString(resultado_4));
    exit(1);
  }

  // aloca para device
  resultado_1 = cudaMalloc((void **)&dev_s, sizeof(char) * (n + 1));
  resultado_2 = cudaMalloc((void **)&dev_r, sizeof(char) * (m + 1));
  resultado_3 = cudaMalloc((void **)&dev_dist, sizeof(int) * (n + 1) * (m + 1));

  if (resultado_1 != cudaSuccess)
  {
    printf("%s\n", cudaGetErrorString(resultado_1));
    exit(1);
  } else if (resultado_2 != cudaSuccess)
  {
    printf("%s\n", cudaGetErrorString(resultado_2));
    exit(1);
  } else if (resultado_3 != cudaSuccess)
  {
    printf("%s\n", cudaGetErrorString(resultado_3));
    exit(1);
  }
}

void libera()
{
  cudaFree(dev_s);
  cudaFree(dev_r);
  cudaFree(dev_dist);
  cudaFreeHost(s);
  cudaFreeHost(r);
  cudaFreeHost(dist);
  cudaFreeHost(dist_res);
}


void imprime_sequencias()
{
  int i;
  for(i=0; i<=n; i++)
  {
    printf("%c ", s[i]);  
  }
  printf("\n");
  for(i=0; i<=m; i++)
  {
    printf("%c ", r[i]);  
  }
  printf("\n");
}

void imprime_matriz()
{
  int i, j;
  printf("  ");
  for(i=0; i<=m; i++)
  {
    printf("%c ", r[i]);  
  }
  printf("\n");
  for(i=0; i<=n; i++)
  {
    printf("%c ", s[i]);
    for(j=0; j<=m; j++)
    {
      printf("%d ", dist[i*(m + 1)+j]);
    }
    printf("\n");
  }
}

void imprime_matriz_resposta()
{
  int i, j;
  for(i=0; i<=n; i++)
  {
    for(j=0; j<=m; j++)
    {
      printf("%d ", dist_res[i*(m + 1)+j]);
    }
    printf("\n");
  }
}


void inicializa(char* entrada)
{
  int i;
  // abre arquivo de entrada
  FILE *arq;
  arq = fopen(entrada, "rt");

  if (arq == NULL)
	{
		printf("\nArquivo texto de entrada não encontrado\n") ;
		exit(1) ;
	}

  // lê tamanho das sequências s e r
  fscanf(arq, "%d %d", &n, &m);
  //printf("entrada %d %d\n", n, m);

  // aloca sequencias e matrizes
  aloca();

  //inicializa sequencias
  s[0] = ' '; 
  r[0] = ' ';
  fscanf(arq, "%s", &(s[1]));
	fscanf(arq, "%s", &(r[1]));

  // fecha arquivo de entrada
  fclose(arq);

  // inicializa valores na matriz
  for(i=1; i<=n; i++) dist[i*(m+1)] = i;
  for(i=0; i<=m; i++) dist[i] = i;

  //printf("imprime sequencias\n");
  //imprime_sequencias();
  //printf("imprime matriz\n");
  //imprime_matriz();

  cudaMemcpy(dev_s, s, sizeof(char)*(n+1), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_r, r, sizeof(char)*(m+1), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_dist, dist, sizeof(int)*(n+1)*(m+1), cudaMemcpyHostToDevice);

}

__global__ 
void distancia_antidiagonal(int* aux, int passos, int n, int m, char* seq_s, char* seq_r) {
	int i = n - threadIdx.x - (blockIdx.x * 512);
	int j = passos - i;

	if(j>0 && j<=m && i>0)
  {
    int norte, noroeste, oeste, idx, t;

    idx = i*(m+1)+j;

    t = seq_s[i] == seq_r[j]? 0 : 1;
    norte = aux[idx-m-1] + 1;
    oeste = aux[idx-1] + 1;
    noroeste = aux[idx-m-2] + t;

    aux[idx] = min(norte, min(noroeste, oeste));
  }
}

int main(int argc, char **argv)
{

  if(argc != 2)
	{
		printf("O programa foi executado com argumentos incorretos.\n") ;
		printf("Uso: ./dist_seq <nome arquivo entrada>\n") ;
		exit(1) ;
	}

  // inicializa variáveis
  inicializa(argv[1]);

  // cria variáveis de evento
  cudaEvent_t inicio, fim;

  // cria eventos
  cudaEventCreate(&inicio);
  cudaEventCreate(&fim);

  cudaEventRecord(inicio, 0);
  
  // numero de antidiagonais
  int n_ad = n+m+1;
  // dimensao grid
  int n_blocks = ceil((double) n / 512);
  n_blocks = n_blocks == 0? 1: n_blocks;
  // numero de passos na antidiagonais
  int passos;

  for (passos=2; passos <= n_ad; passos++) {
    // calcular grid e blocos
    dim3 dimGrid(n_blocks);
    dim3 dimBlock(512);

    // calcula distancia na anti-diagonal
    distancia_antidiagonal<<<dimGrid,dimBlock>>>(dev_dist, passos, n, m, dev_s, dev_r);
  }

  // copia resposta de device para host
  cudaMemcpy(dist_res, dev_dist, sizeof(int)*(n+1)*(m+1), cudaMemcpyDeviceToHost);

  //imprime_matriz_resposta();
  printf("Distância=%d\n", dist_res[((n + 1) * (m + 1))-1]);

  cudaEventRecord(fim, 0);
  cudaEventSynchronize(fim);

  float tempo = 0;
  cudaEventElapsedTime(&tempo, inicio, fim);

  // destrói eventos
  cudaEventDestroy(inicio);
  cudaEventDestroy(fim);

  // libera sequencias e matrizes
  libera();

  printf("Tempo GPU = %.2fms\n", tempo);

  return 0;
}
