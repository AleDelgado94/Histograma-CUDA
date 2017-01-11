/*******************************
 * Autor: Alejandro Delgado Martel
 * Nombre: Proyecto Final Versión 1
 *******************************/


#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

#define HISTO_ELEMENTS 1000


__global__ void inicializa_histograma(float* A, float* hist, int num_elements, int hist_elements){
	int i = (blockIdx.x * blockDim.x + threadIdx.x);


	if(i < num_elements){
		if(i==0){
			for(int j=0; j<hist_elements; j++)
				hist[j]=0.0;
		}
	}
}

__global__ void histograma(float* A, float* hist, int num_elements, int hist_elements){


	//Posicion del thread
	int i = (blockIdx.x * blockDim.x + threadIdx.x);


	if(i < num_elements){

		int pos = (int)(fmod(A[i],(float)hist_elements));
		atomicAdd(&(hist[pos]), 1.0);
	}
}



void fError(cudaError_t err){
	if(err != cudaSuccess){
		printf("Ha ocurrido un error el la linea %d con codigo: %s\n", __LINE__, cudaGetErrorString(err));
	}
}


int main(){

	//cudaSetDevice(0);

	int num_elements = 1000000;
	int hist_elements = HISTO_ELEMENTS;

	//Reservar espacio en memoria HOST


	float * h_A = (float*)malloc(num_elements * sizeof(float));

	float * h_hist = (float*)malloc(hist_elements * sizeof(float));


	if(h_A == NULL ){
		printf("Error al reservar memoria para los vectores HOST");
		exit(1);
	}



	//Inicializar elementos de los vectores de forma hormogenea
	for(int i=0; i<num_elements; i++){
		h_A[i] = (float)i;
	}

	//Inicializamos a 0 todas las posiciones del histograma
	//for(int i=0; i<hist_elements; i++)
		//h_hist[i]=0.0;


	cudaError_t err;

	int size = num_elements * sizeof(float);
	int size_hist = hist_elements * sizeof(float);


	float * d_A = NULL;
	err = cudaMalloc((void **)&d_A, size);
	fError(err);

	float * d_hist = NULL;
	err = cudaMalloc((void**)&d_hist, size_hist);
	fError(err);


	//Copiamos a GPU DEVICE
	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	err = cudaMemcpy(d_hist, h_hist, size_hist, cudaMemcpyHostToDevice);


	int HilosPorBloque = 512;
	int BloquesPorGrid = (num_elements + HilosPorBloque -1) / HilosPorBloque;


	cudaError_t Err;

	//Lanzamos el kernel y medimos tiempos
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	inicializa_histograma<<<BloquesPorGrid, HilosPorBloque>>>(d_A, d_hist, num_elements, hist_elements);
	histograma<<<BloquesPorGrid, HilosPorBloque>>>(d_A,d_hist, num_elements, hist_elements);


	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
	float tiempo_reserva_host;
	cudaEventElapsedTime(&tiempo_reserva_host, start, stop);

	Err = cudaGetLastError();
	fError(Err);


	printf("Tiempo de suma vectores DEVICE: %f\n", tiempo_reserva_host);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	//Copiamos a CPU el vector C
	err = cudaMemcpy(h_A, d_A, size, cudaMemcpyDeviceToHost);
	fError(err);

	err = cudaMemcpy(h_hist, d_hist, size_hist, cudaMemcpyDeviceToHost);
	fError(err);



	float suma = 0;

	for(int i=0; i<hist_elements; i++){
		//printf("%f \n", h_hist[i]);
		//printf("\n");
		suma = suma + h_hist[i];
	}

	printf("La suma es: %f", suma);

}

