/*
	CS 6023 Assignment 3. 
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>

#define DEBUG 0
#define TIME_CHECK 0
#define BLOCKSIZE 1024

void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input. 
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ;
	

	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ;
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ;
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ;
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
			}
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ; 
		scenes.push_back (scene) ;
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ;
	int u, v ; 
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL; 
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}
	
	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}


__global__
void translate_all(int *gGlobalCoordinatesX, int * gGlobalCoordinatesY, int *gHorizontal, int * gVertical, int N){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= N) return;
	if(DEBUG) printf("Id: %d\t Coordinates: %d %d\t Shift: %d %d\n", id, gGlobalCoordinatesX[id], gGlobalCoordinatesY[id], gHorizontal[id], gVertical[id]);
    gGlobalCoordinatesX[id] += gVertical[id];
    gGlobalCoordinatesY[id] += gHorizontal[id];
}

__global__
void fill_translations(int *gTranslations, int *gHorizontal, int * gVertical, int T){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id >= 3*T || id%3 != 0) return;
	int frame_no = gTranslations[id];
	int direction = gTranslations[id+1];
	int amount = gTranslations[id+2];
	if(direction == 0) atomicSub(&gVertical[frame_no], amount);
	if(direction == 1) atomicAdd(&gVertical[frame_no], amount);
	if(direction == 2) atomicSub(&gHorizontal[frame_no], amount);
	if(direction == 3) atomicAdd(&gHorizontal[frame_no], amount);
}

__global__
void generate_image(int **gMesh, int *gGlobalCoordinatesX, int *gGlobalCoordinatesY, int *gFinalPng, int *gFrameSizeX, int *gFrameSizeY, int *gOpacity, int M, int N, int V){
	int id = threadIdx.x + (blockDim.x) * blockIdx.x;
	int img_row = id / M;
	int img_col = id % M;

	if(img_row >= N || img_col >= M) return;
	int max_opacity = -1;
	int value_to_write = 0;
	for(int i = 0; i < V; i++){
		int N_i = gFrameSizeX[i];
		int M_i = gFrameSizeY[i];
		int i_row = img_row - gGlobalCoordinatesX[i];
		int i_col = img_col - gGlobalCoordinatesY[i];
		if(i_row < 0 || i_col < 0 || i_row >= N_i || i_col >= M_i) continue;
		if(gOpacity[i] > max_opacity){
			max_opacity = gOpacity[i];
			value_to_write = gMesh[i][i_row * M_i + i_col];
		}
	}
	gFinalPng[id] = value_to_write;
}

__global__
void bfs_helper(int *curr_queue, int *next_queue, int *gHorizontal, int * gVertical, int *gCsr, int *gOffset, int V, int curr_queue_size, int *next_queue_size) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    if (id >= curr_queue_size) return;
    int node = curr_queue[id];
    for (int i = gOffset[node]; i < gOffset[node + 1]; i++) {
        atomicAdd(&gVertical[gCsr[i]], gVertical[node]);
        atomicAdd(&gHorizontal[gCsr[i]], gHorizontal[node]);
        int next_index = atomicAdd(next_queue_size, 1);
        next_queue[next_index] = gCsr[i];
    }
}

void fill_bfs(int *gHorizontal, int * gVertical, int *gCsr, int *gOffset, int V) {
    int *curr_queue;
    cudaMalloc(&curr_queue, V * sizeof(int));
    cudaMemset(curr_queue, 0, V * sizeof(int));
    int curr_queue_size = 1;
    while (curr_queue_size > 0) {
        int *next_queue_size;
        cudaMalloc((void **)&next_queue_size, sizeof(int));
        cudaMemset(next_queue_size, 0, sizeof(int));

        int *next_queue;
        cudaMalloc((void **)&next_queue, V * sizeof(int));

        int num_blocks = (curr_queue_size + BLOCKSIZE - 1) / BLOCKSIZE;
        int num_threads = min(BLOCKSIZE, curr_queue_size);
        bfs_helper<<<num_blocks, num_threads>>>(curr_queue, next_queue, gHorizontal, gVertical, gCsr, gOffset, V, curr_queue_size, next_queue_size);
        cudaDeviceSynchronize(); 
        cudaMemcpy(&curr_queue_size, next_queue_size, sizeof(int), cudaMemcpyDeviceToHost);
        if (curr_queue_size == 0)
            break; // Exit loop if queue is empty
        cudaMemcpy(curr_queue, next_queue, curr_queue_size * sizeof(int), cudaMemcpyDeviceToDevice);
        cudaFree(next_queue_size);
        cudaFree(next_queue);
    }
}

int main (int argc, char **argv) {
	
	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng ; 

	int frameSizeX, frameSizeY ;
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;
	
	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ;
	int E = edges.size () ;
	int numTranslations = translations.size () ;

	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset () ;  
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now () ;
	// Code begins here.
	// Do not change anything above this comment
	int num_blocks, num_threads; // declaring in advance

	/* Translation :: Sequential */
	int *hTranslations = (int *) malloc(3 * numTranslations * sizeof(int));
	int htrans_id = 0;
    for(auto &translate: translations){
		hTranslations[htrans_id++] = translate[0];
		hTranslations[htrans_id++] = translate[1];
		hTranslations[htrans_id++] = translate[2];
	}

    auto checkpoint1_1  = std::chrono::high_resolution_clock::now () ;
	std::chrono::duration<double, std::micro> translation_sequential_time = checkpoint1_1-start;
	if(TIME_CHECK) printf ("sequential translation time : %f\n", translation_sequential_time) ;

	/* Memory Allocation Starts */
    int * gGlobalCoordinatesX;
    int * gGlobalCoordinatesY;
	int * gOpacity;
    int * gFrameSizeX;
    int * gFrameSizeY;
	int * gFinalPng;
	int * gCsr;
	int * gOffset;
	int **gMesh;

	cudaMalloc(&gGlobalCoordinatesX, V * sizeof(int));
    cudaMalloc(&gGlobalCoordinatesY, V * sizeof(int));
	cudaMalloc(&gCsr, E * sizeof(int));
    cudaMalloc(&gOffset, (V+1) * sizeof(int));

	cudaMalloc(&gOpacity, V * sizeof(int));
	cudaMalloc(&gFrameSizeX, V * sizeof(int));
	cudaMalloc(&gFrameSizeY, V * sizeof(int));
	cudaMalloc(&gFinalPng, frameSizeX * frameSizeY * sizeof(int));
	cudaMalloc((void **) &gMesh, V * sizeof(int *));

    cudaMemcpy(gGlobalCoordinatesX, hGlobalCoordinatesX, V * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gGlobalCoordinatesY, hGlobalCoordinatesY, V * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gCsr, hCsr, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gOffset, hOffset, (V+1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gOpacity, hOpacity, V * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gFrameSizeX, hFrameSizeX, V * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gFrameSizeY, hFrameSizeY, V * sizeof(int), cudaMemcpyHostToDevice);	
    for (int i = 0; i < V; ++i) {
		int * gMesh_i;
		cudaMalloc(&gMesh_i, hFrameSizeX[i] * hFrameSizeY[i] * sizeof(int));
		cudaMemcpy(gMesh_i, hMesh[i], hFrameSizeX[i] * hFrameSizeY[i] * sizeof(int), cudaMemcpyHostToDevice);

		// store this in the ith position of gMesh
		cudaMemcpy(&gMesh[i], &gMesh_i, sizeof(int *), cudaMemcpyHostToDevice);
    }

	/* Extra Created Arrays */
	int * gHorizontal;
    int * gVertical;
	int * gTranslations;
    cudaMalloc(&gHorizontal, V * sizeof(int));
    cudaMalloc(&gVertical, V * sizeof(int));
    cudaMalloc(&gTranslations, 3 * numTranslations * sizeof(int));
	cudaMemcpy(gTranslations, hTranslations, 3 * numTranslations * sizeof(int), cudaMemcpyHostToDevice);

	/* Memory Allocation Done */
    auto checkpoint1_2 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::micro> total_memory_allocation_time = checkpoint1_2-checkpoint1_1;
	if(TIME_CHECK) printf ("total memory allocation time : %f\n", total_memory_allocation_time) ;

	num_blocks = (3*numTranslations + BLOCKSIZE - 1) / BLOCKSIZE;
	num_threads = min(3*numTranslations, BLOCKSIZE);

	/* Kernel Call */
	fill_translations<<<num_blocks, num_threads>>> (gTranslations, gHorizontal, gVertical, numTranslations);
	cudaDeviceSynchronize();

	/* Deallocation of Memory */
	cudaFree(gTranslations);
	free(hTranslations);

	auto checkpoint2_1 = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::micro> translation_fill_time = checkpoint2_1-checkpoint1_2;
	if(TIME_CHECK) printf ("translation fill time : %f\n", translation_fill_time) ;

	/* Parallelized BFS Function Call */
	fill_bfs(gHorizontal, gVertical, gCsr, gOffset, V);

    auto checkpoint2_2  = std::chrono::high_resolution_clock::now () ;
	std::chrono::duration<double, std::micro> parallelized_traversal = checkpoint2_2-checkpoint2_1;
	if(TIME_CHECK) printf("parallelized traversal time: %f\n", parallelized_traversal);

    num_blocks = (V + BLOCKSIZE - 1) / BLOCKSIZE;
    num_threads = min(BLOCKSIZE, V);

	/* Kernel Call */
    translate_all<<<num_blocks, num_threads>>> (gGlobalCoordinatesX, gGlobalCoordinatesY, gHorizontal, gVertical, V);
    cudaDeviceSynchronize(); // Need to Synchronize
		
    /* Deallocation of memory */
    cudaFree(gHorizontal);
    cudaFree(gVertical);

    auto checkpoint2_3  = std::chrono::high_resolution_clock::now () ;
	std::chrono::duration<double, std::micro> total_translation_time = checkpoint2_3-checkpoint2_2;
	if(TIME_CHECK) printf ("total translation time : %f\n", total_translation_time) ;

	num_blocks = (frameSizeX * frameSizeY + BLOCKSIZE - 1) / BLOCKSIZE;
	num_threads = min(frameSizeX * frameSizeY, BLOCKSIZE);

	/* Kernel Call */
	generate_image<<<num_blocks, num_threads>>>(gMesh, gGlobalCoordinatesX, gGlobalCoordinatesY, gFinalPng, gFrameSizeX, gFrameSizeY, gOpacity, frameSizeX, frameSizeY, V);
    cudaDeviceSynchronize();

	/* Getting the final image back to CPU */
	cudaMemcpy(hFinalPng, gFinalPng, frameSizeX * frameSizeY * sizeof(int), cudaMemcpyDeviceToHost);

    auto checkpoint3_1  = std::chrono::high_resolution_clock::now () ;
	std::chrono::duration<double, std::micro> image_generation_time = checkpoint3_1-checkpoint2_3;
	if(TIME_CHECK) printf ("image generation time : %f\n", image_generation_time) ;

	// Do not change anything below this comment.
	// Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;	

}
