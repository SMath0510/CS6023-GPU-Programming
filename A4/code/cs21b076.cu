#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <chrono>

using namespace std;

#define DEBUG 0
#define TIMECHECK 0
#define BLOCKSIZE 1024
//*******************************************

// Write down the kernels here
__global__
void deal_with_tanks(int *g_xcoord, int *g_ycoord, int* g_hp, int *g_hp_mod, int *g_score, int T, long long int round_num, int *num_tanks){
    extern __shared__ int s_data[];
    int *s_xcoord = &s_data[0];
    int *s_ycoord = &s_data[T];
    int *s_hp = &s_data[2 * T];
    
    int curr_tank_id = threadIdx.x + blockDim.x * blockIdx.x;
    if(curr_tank_id >= T) return; // Out of Index Access OR Null Round
    // printf("Round Num: %d\n", round_num);
    g_hp[curr_tank_id] = g_hp_mod[curr_tank_id];
    if(g_hp[curr_tank_id] > 0) atomicAdd(num_tanks, 1);
    if(round_num %T == 0) return;
    __syncthreads();

    /* Load data into shared memory */
    s_xcoord[curr_tank_id] = g_xcoord[curr_tank_id];
    s_ycoord[curr_tank_id] = g_ycoord[curr_tank_id];
    s_hp[curr_tank_id] = g_hp[curr_tank_id];

    __syncthreads();

    if(s_hp[curr_tank_id] <= 0) return; // Dead
    
    int next_tank_id = (curr_tank_id + round_num) % T; 
    long long int direction_x = s_xcoord[next_tank_id] - s_xcoord[curr_tank_id];
    long long int direction_y = s_ycoord[next_tank_id] - s_ycoord[curr_tank_id];
    long long int min_dist = LONG_LONG_MAX;
    long long int hit_tank_id = -1;

    for(int tank_id = 0; tank_id < T; tank_id ++){
        if(tank_id == curr_tank_id || s_hp[tank_id] <= 0) continue; // Same or Dead
        long long int other_direction_x = s_xcoord[tank_id] - s_xcoord[curr_tank_id];
        long long int other_direction_y = s_ycoord[tank_id] - s_ycoord[curr_tank_id];
        bool is_same_slope = (other_direction_x * direction_y) == (other_direction_y * direction_x);
        bool is_same_direction = ((other_direction_x * direction_x) >= 0) && ((other_direction_y * direction_y) >= 0);
        if (is_same_slope && is_same_direction){
            long long int other_dist = (other_direction_x * other_direction_x + other_direction_y * other_direction_y);
            if(DEBUG){
                printf("Target Tank for Tank %d is Tank %d in Round %ld\n", curr_tank_id, tank_id, round_num);
                printf("Original Slope value: %lld/%lld; New Slope value: %lld/%lld\n", direction_y, direction_x, other_direction_y, other_direction_x);
            }
            if(other_dist <= min_dist){
                min_dist = other_dist;
                hit_tank_id = tank_id;
            }
        }
    }
    
    if(hit_tank_id != -1){
        g_score[curr_tank_id] ++;
        atomicSub(&g_hp_mod[hit_tank_id], 1);
        if(DEBUG) printf("%d HIT %d in Round %ld\n", curr_tank_id, hit_tank_id, round_num);
    }
}


__global__
void initialize_device_values(int *g_hp_mod, int *g_hp, int *g_score, int *num_tanks, int H, int T){
    long long int tank_id = threadIdx.x + blockDim.x * blockIdx.x;
    if(tank_id >= T) return;
    /*
        Initialize HP arrays to value H each
        Initialize Score to 0
        Set numtanks to 0
    */
    g_hp_mod[tank_id] = H;
    g_hp[tank_id] = H;
    g_score[tank_id] = 0;
}

//***********************************************

int main(int argc,char **argv)
{
    // Variable declarations
    int M,N,T,H,*xcoord,*ycoord,*score;
    

    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");

    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &M );
    fscanf( inputfilepointer, "%d", &N );
    fscanf( inputfilepointer, "%d", &T ); // T is number of Tanks
    fscanf( inputfilepointer, "%d", &H ); // H is the starting Health point of each Tank
	
    // Allocate memory on CPU
    xcoord=(int*)malloc(T * sizeof (int));  // X coordinate of each tank
    ycoord=(int*)malloc(T * sizeof (int));  // Y coordinate of each tank
    score=(int*)malloc(T * sizeof (int));  // Score of each tank (ensure that at the end you have copied back the score calculations on the GPU back to this allocation)

    // Get the Input of Tank coordinates
    for(int i=0;i<T;i++)
    {
      fscanf( inputfilepointer, "%d", &xcoord[i] );
      fscanf( inputfilepointer, "%d", &ycoord[i] );
    }
		

    auto start = chrono::high_resolution_clock::now();

    //*********************************
    // Your Code begins here (Do not change anything in main() above this comment)
    //********************************
    
    /* 
        hp_mod -> stores the HP after the round. // workaround for using global barriers
    */

    // Memory Allocation
    int *g_hp, *g_hp_mod, *g_score, *g_xcoord, *g_ycoord, *num_tanks;
    cudaMalloc(&g_hp, T * sizeof(int));
    cudaMalloc(&g_hp_mod, T * sizeof(int));
    cudaMalloc(&g_score, T * sizeof(int));
    cudaMalloc(&g_xcoord, T * sizeof(int));
    cudaMalloc(&g_ycoord, T * sizeof(int));
    cudaMalloc(&num_tanks, sizeof(int));
    cudaMemcpy(g_xcoord, xcoord, T * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(g_ycoord, ycoord, T * sizeof(int), cudaMemcpyHostToDevice);

    auto check_point_1  = chrono::high_resolution_clock::now();
    chrono::duration<double, std::micro> memory_allocation_time = check_point_1-start;
    if(TIMECHECK) printf("memory allocation time : %f\n", memory_allocation_time.count());

    // Initialize
    int num_blocks = (T + BLOCKSIZE - 1)/ BLOCKSIZE;
    initialize_device_values<<<num_blocks, BLOCKSIZE>>>(g_hp_mod, g_hp, g_score, num_tanks, H, T);
    cudaDeviceSynchronize();

    auto check_point_2  = chrono::high_resolution_clock::now();
    chrono::duration<double, std::micro> initialize_time = check_point_2-check_point_1;
    if(TIMECHECK) printf("initialize time : %f\n", initialize_time.count());

    long long int round_num = 1;
    while(1){
        cudaMemset(num_tanks, 0, sizeof(int));

        deal_with_tanks<<<num_blocks, BLOCKSIZE, 3 * T * sizeof(int)>>>(g_xcoord, g_ycoord, g_hp, g_hp_mod, g_score, T, round_num, num_tanks);
        cudaDeviceSynchronize();
    
        
        int *curr_num_tanks = (int *) malloc(sizeof(int));
        cudaMemcpy(curr_num_tanks, num_tanks, sizeof(int), cudaMemcpyDeviceToHost);
        if(*curr_num_tanks <= 1) break;

        round_num ++;
    }

    cudaMemcpy(score, g_score, T * sizeof(int), cudaMemcpyDeviceToHost);

    //*********************************
    // Your Code ends here (Do not change anything in main() below this comment)
    //********************************

    auto end  = chrono::high_resolution_clock::now();
    chrono::duration<double, std::micro> timeTaken = end-start;
    printf("Execution time : %f\n", timeTaken.count());

    // Output
    char *outputfilename = argv[2];
    char *exectimefilename = argv[3]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    for(int i=0;i<T;i++)
    {
        fprintf( outputfilepointer, "%d\n", score[i]);
    }
    fclose(inputfilepointer);
    fclose(outputfilepointer);

    outputfilepointer = fopen(exectimefilename,"w");
    fprintf(outputfilepointer,"%f", timeTaken.count());
    fclose(outputfilepointer);

    free(xcoord);
    free(ycoord);
    free(score);
    cudaDeviceSynchronize();
    return 0;
}