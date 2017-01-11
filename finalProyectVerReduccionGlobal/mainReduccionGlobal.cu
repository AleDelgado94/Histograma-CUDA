/*******************************
 * Autor: Alejandro Delgado Martel
 * Nombre: Proyecto Final Versión 2 Reducción memoria SHARED
 *
 *
 *
 *
 * 	NOTA: mirar extern __shared__ variables
 *
 *
 *******************************/


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

#include <iostream>

using namespace std;

#define HISTO_ELEMENTS 1000


__global__ void inicializa_histograma(int num_elements, const int hist_elements, float* hist_locales){

	int i = (blockIdx.x * blockDim.x + threadIdx.x);


	if(i < num_elements){
		if(threadIdx.x == 0){
			for(int k=0;k<hist_elements;k++){
				hist_locales[blockIdx.x*hist_elements+k] = 0.0;
			}

		}
	}
}


__global__ void histograma(float* A, float* hist, int num_elements, const int hist_elements, int nBloques){


	//Posicion del thread
	int i = (blockIdx.x * blockDim.x + threadIdx.x);

	if(i < num_elements){

		//CALCULAMOS LOS HISTOGRAMAS APLICANDO LA SIGUIENTE FORMULA
		int pos = (int)(fmod(A[i],(float)hist_elements));
		atomicAdd(&(hist[(blockIdx.x * HISTO_ELEMENTS) + pos]), 1.0);

		//ESPERAMOS A QUE LOS HILOS DEL MISMO BLOQUE TERMINEN
		__syncthreads();
	}


}


__global__ void reduccion(float* hist, float* hist_reducido, int num_elements, const int hist_elements, int num_bloques){

	int i = (blockIdx.x * blockDim.x + threadIdx.x);

	//UNA VEZ TENEMOS CREADOS LOS HISTOGRAMAS LOCALES A CADA BLOQUE EN UNA MATRIZ UNIDIMENSIONAL,
	//EMPLEAREMOS EL MÉTODO DE LA SUMA POR REDUCCIÓN Y COPIAREMOS EL RESULTADO EN EL HISTOGRAMA
	//FINAL

	//NOS TENEMOS QUE ASEGURAR QUE EL NUMERO DE BLOQUES DEL GRID SEA POTENCIA DE DOS PARA QUE
	//EL MÉTODO DE LA REDUCCIÓN FUNCIONE CORRECTAMENTE

	//SUMA POR METODO REDUCCIÓN
	if(i < (num_bloques*HISTO_ELEMENTS)/2){
		//VAMOS REALIZANDO LA SUMA DE LA PRIMERA MITAD CON LA SEGUNDA Y SE
		//ALMACENA EN LA PRIMERA
		atomicAdd(&(hist[i]), hist[ (num_bloques/2) * HISTO_ELEMENTS + i ]);
	}




	//COPIAMOS EL VALOR DEL HISTOGRAMA (MATRIZ UNIDIMENSIONAL) AL HISTOGRAMA FINAL
	if(i < HISTO_ELEMENTS){
		hist_reducido[i] = hist[i];
		__syncthreads();
	}

}



void fError(cudaError_t err, int linea){
	if(err != cudaSuccess){
		printf("Ha ocurrido un error el la linea %d con codigo: %s\n", linea, cudaGetErrorString(err));
	}
}


int main(){

	//cudaSetDevice(0);

	int num_elements = 1000000;
	int hist_elements = HISTO_ELEMENTS;

	int HilosPorBloque = 977;
	int BloquesPorGrid = (num_elements + HilosPorBloque -1) / HilosPorBloque;
	int nBloques = BloquesPorGrid;


	//Reservar espacio en memoria HOST


	float * h_A = (float*)malloc(num_elements * sizeof(float));

	float * h_hist = (float*)malloc(hist_elements * sizeof(float));

	float * h_Histo = (float*)malloc(BloquesPorGrid * hist_elements * sizeof(float));

	float * h_Reducido = (float*)malloc(hist_elements * sizeof(float));

	if(h_A == NULL || h_hist == NULL || h_Histo == NULL){
		printf("Error al reservar memoria para los vectores HOST");
		exit(1);
	}



	//Inicializar elementos de los vectores de forma hormogenea
	for(int i=0; i<num_elements; i++){
		h_A[i] = (float)i;
	}

	for(int i=0; i<hist_elements; i++){
		h_Reducido[i] = 0.0;
	}

	cudaError_t err;

	int size = num_elements * sizeof(float);
	int size_hist = hist_elements * sizeof(float);
	int size_Histo = BloquesPorGrid * hist_elements * sizeof(float);

	float * d_A = NULL;
	err = cudaMalloc((void **)&d_A, size);
	fError(err,__LINE__);

	float * d_hist = NULL;
	err = cudaMalloc((void**)&d_hist, size_hist);
	fError(err,__LINE__);

	//Array que almacena los histogramas locales
	float * d_Histo = NULL;
	err = cudaMalloc((void**)&d_Histo, size_Histo);

	float* d_Reducido = NULL;
	err = cudaMalloc((void**)&d_Reducido, size_hist);



	//Copiamos a GPU DEVICE
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	fError(err,__LINE__);
	err = cudaMemcpy(d_hist, h_hist, size_hist, cudaMemcpyHostToDevice);
	fError(err,__LINE__);
	err = cudaMemcpy(d_Reducido, h_Reducido, size_hist, cudaMemcpyHostToDevice);
	fError(err, __LINE__);



	printf("Numero de bloques: %d\n", BloquesPorGrid);
	printf("Numero de hilos por bloque: %d\n", HilosPorBloque);
	printf("Tamaño del histograma: %d\n", HISTO_ELEMENTS);



/**********************EJECUTANDO KERNELS***************************/

	cudaError_t Err;

	//Lanzamos el kernel y medimos tiempos
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	inicializa_histograma<<<BloquesPorGrid, HilosPorBloque>>>(num_elements, hist_elements, d_Histo);
	histograma<<<BloquesPorGrid, HilosPorBloque>>>(d_A, d_Histo, num_elements, hist_elements, BloquesPorGrid);

	//EJECUTAMOS EL KERNEL REDUCE TANTAS VECES COMO TAMAÑO DEL NÚMERO DE
	//BLOQUES QUE TENGAMOS ES POR ELLO QUE EJECUTAMOS log2(Numero_de_bloques)
	//Y POSTERIORMENTE VAMOS REDUCIENDO ESE NÚMERO A LA MITAD
	for(int i=0; i<log2((double)nBloques); i++){
		reduccion<<<BloquesPorGrid, HilosPorBloque>>>(d_Histo, d_Reducido, num_elements, hist_elements, BloquesPorGrid);
		BloquesPorGrid /= 2;
	}


	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float tiempo_reserva_host;
	cudaEventElapsedTime(&tiempo_reserva_host, start, stop);

	Err = cudaGetLastError();
	fError(Err,__LINE__);



	printf("Tiempo de suma vectores DEVICE: %f\n", tiempo_reserva_host);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);



/***********************************************************************/


	//Copiamos a CPU el vector C
	err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
	fError(err,__LINE__);

	err = cudaMemcpy(h_hist, d_hist, size_hist, cudaMemcpyDeviceToHost);
	fError(err,__LINE__);

	err = cudaMemcpy(h_Reducido, d_Reducido, size_hist, cudaMemcpyDeviceToHost);
	fError(err, __LINE__);

	err = cudaMemcpy(h_Histo, d_Histo, size_Histo, cudaMemcpyDeviceToHost);
	fError(err, __LINE__);


/*******************COMPROBANDO RESULTADOS ****************************/
	float suma = 0;


	for(int j=0; j</*BloquesPorGrid * */HISTO_ELEMENTS; j++){
		//printf("%f \n", h_Reducido[j]);
		//printf("\n");
		suma = suma + h_Reducido[j];
	}

	printf("La suma total es: %f\n", suma);
	printf("Con un tamaño de %d\n", BloquesPorGrid*HISTO_ELEMENTS);


/***********************************************************************/


	err = cudaFree(d_A);
	    if (err != cudaSuccess)
	    {
	        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }

	err = cudaFree(d_hist);

	    if (err != cudaSuccess)
	    {
	        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }

	err = cudaFree(d_Histo);

	    if (err != cudaSuccess)
	    {
	        fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
	        exit(EXIT_FAILURE);
	    }


	err = cudaFree(d_Reducido);

			if (err != cudaSuccess)
			{
				fprintf(stderr, "Failed to free device vector C (error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}

	    // Free host memory
	    free(h_A);
	    free(h_hist);
	    free(h_Histo);
	    free(h_Reducido);


}

