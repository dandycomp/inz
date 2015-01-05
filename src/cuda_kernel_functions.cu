#include "cuda_kernel_functions.cuh"
#include <math.h>


__global__ void cudaGetBestCorrelateFromArea(int* p_values, int* glsum1,
	int* glsum2, int* glSqSum1, int* glSqSum2,
	int x, int y, float* outp)
{
	// pozniej trzeba bedzie obliczenie dla pierwszego patcha wyniesc poza funkcje i przekazywac jako argument
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	const int block_side = SUB_IMG - PATCH_SIZE + 1;
	const int distortion = block_side / 2;// przesuniecie poczatku odliczania od punktu x,y

	float f_value = 0.0;
	float g_value = 0.0;
	float q_value = 0.0;

	const int gl_dist = PATCH_SIZE / 2;
	int y_ = ty - block_side / 2 - gl_dist;// przesuniecie punktu poczatku odliczania wzgledem x,y
	int x_ = tx - block_side / 2 - gl_dist;

	int gl_width = IMAGE_WIDTH - PATCH_SIZE + 1;// width of global images

	int index1 = (y - gl_dist) * gl_width + (x - gl_dist);
	int index2 = (y + y_)*gl_width + x + x_;

	float glSum1 = (float)glsum1[index1];
	float glSum2 = (float)glsum2[index2];
	float pixelInPatch = (float)PATCH_SIZE *(float)PATCH_SIZE;

	f_value = (float)glSqSum1[index1] - glSum1 * glSum1 / pixelInPatch;// moze byc problem z zaokragleniem
	g_value = (float)glSqSum2[index2] - glSum2 * glSum2 / pixelInPatch;// moze byc problem z zaokragleniem
	q_value = glSum1 * glSum2 / pixelInPatch;
	float p_value = p_values[ty*block_side + tx];

	if (f_value == 0 || g_value == 0)
	{
		//printf("\nbefore = %d", (p_value - q_value) / sqrt(f_value * g_value));
		outp[ty*block_side + tx] = -1.0;
		//printf("\nafter = %f\n", outp[ty*block_side + tx]);
	}
	else
		outp[ty*block_side + tx] = (p_value - q_value) / sqrt(f_value * g_value);
}


__global__ void cudaGetAll_P_FromArea(int *img1, int* img2,
	int x, int y, int* outpPvector)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int bx = blockIdx.x;
	int by = blockIdx.y;

	int block_side = SUB_IMG - PATCH_SIZE + 1;

	int dist_x = bx - block_side / 2;
	int dist_y = by - block_side / 2;

	int p_sum = 0;

	__shared__ int shared_img1[PATCH_SIZE*PATCH_SIZE];
	__shared__ int shared_img2[PATCH_SIZE*PATCH_SIZE];

	int thread_X = x + tx - PATCH_SIZE / 2;// uwzgledniamy i przesuniêcie centralnego piksela oraz caly patch dookola tego piksela
	int thread_Y = y + ty - PATCH_SIZE / 2;

	shared_img1[tx + PATCH_SIZE*ty] = img1[thread_Y*IMAGE_WIDTH + thread_X];
	shared_img2[tx + PATCH_SIZE*ty] = img2[(thread_Y + dist_y)*IMAGE_WIDTH + (thread_X + dist_x)];

	__syncthreads();

	for (int k = 0; k < PATCH_SIZE*PATCH_SIZE; k++)
		p_sum += shared_img1[k] * shared_img2[k];

	// Synchronize to make sure that the preceding
	// computation is done before loading two new
	// sub-matrices of A and B in the next iteration
	__syncthreads();

	outpPvector[by*block_side + bx] = p_sum;
}



__global__ void cudaSimpleGetBestCorrelate(int* img2, int x, int y,
	int* im1_part, float med, float deviat, float* correlate){

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	const int block_side = SUB_IMG - PATCH_SIZE + 1;
	__shared__ int shared_img2[SUB_IMG*SUB_IMG];// copy all area to shared memory

	int thread_X = x + tx - SUB_IMG / 2;// uwzgledniamy i przesuniêcie centralnego piksela oraz caly patch dookola tego piksela
	int thread_Y = y + ty - SUB_IMG / 2;

	shared_img2[tx + SUB_IMG*ty] = img2[thread_Y*IMAGE_WIDTH + thread_X];
	__syncthreads();
	
	if (tx < block_side && ty < block_side)
	{

	int sum = 0;
	float medium = 0.0;
	float deviation = 0.0;

	for (int row = 0; row < PATCH_SIZE; row++)
		for (int col = 0; col < PATCH_SIZE; col++)
		{
		int x_start = tx + col;//zero in 0,0, not x,y
		int y_start = ty + row;
		sum += shared_img2[x_start + (y_start)*SUB_IMG];
		}
	
	medium = float(sum) / ((float)PATCH_SIZE * (float)PATCH_SIZE);
	
	for (int row = 0; row < PATCH_SIZE; row++)
		for (int col = 0; col < PATCH_SIZE; col++)
		{
		int x_start = tx + col;//zero in 0,0, not x,y
		int y_start = ty + row;
		deviation += pow(shared_img2[x_start + (y_start)*SUB_IMG] - medium, 2);
		}

	float numerator = 0.0;

	for (int row = 0; row < PATCH_SIZE; row++)
		for (int col = 0; col < PATCH_SIZE; col++)
		{
		int x_start = tx + col;//zero in 0,0, not x,y
		int y_start = ty + row;
		numerator += ((float)shared_img2[x_start + (y_start)*SUB_IMG] - medium)*
			((float)im1_part[col + row*PATCH_SIZE] - med);
		}
	float correlation = numerator / sqrt(deviat*deviation);
	//printf(" \n Dla %d, %d   Uzyskana korrelacja = %f", tx, ty, correlation);
	correlate[tx + ty*block_side] = correlation;
	}

}


__global__ void cudaGetBestCorrelate_fast(int* img2, int* glSum2, int* glSqSum2,
							int* im1_part, int x, int y, 
							int glSum1_val, int glSqSum1_val, float* correlate)
{
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	const int block_side = SUB_IMG - PATCH_SIZE + 1;
	__shared__ int shared_img2[SUB_IMG*SUB_IMG];// copy all area to shared memory

	int thread_X = x + tx - SUB_IMG / 2;// uwzgledniamy i przesuniêcie centralnego piksela oraz caly patch dookola tego piksela
	int thread_Y = y + ty - SUB_IMG / 2;

	shared_img2[tx + SUB_IMG*ty] = img2[thread_Y*IMAGE_WIDTH + thread_X];
	__syncthreads();

	if (tx < block_side && ty < block_side)
	{
		int  glX_ = x - block_side / 2 + tx - PATCH_SIZE / 2;
		int  glY_ = y - block_side / 2 + ty - PATCH_SIZE / 2;
		int glWidth = IMAGE_WIDTH - PATCH_SIZE + 1;
		int index = glX_ + glY_*glWidth;
		int glSum2_val = glSum2[index];
		int glSqSum2_val = glSqSum2[index];
		int p_val = 0;

		for (int row = 0; row < PATCH_SIZE; row++)
			for (int col = 0; col < PATCH_SIZE; col++)
			{
			int x_start = tx + col;//zero in 0,0, not x,y
			int y_start = ty + row;
			p_val += shared_img2[x_start + (y_start)*SUB_IMG] *
				im1_part[col + row*PATCH_SIZE];
			}

		float q_val = (float)glSum1_val*(float)glSum2_val / ((float)PATCH_SIZE*(float)PATCH_SIZE);
		float f_val = (float)glSqSum1_val - powf(glSum1_val, 2) / ((float)PATCH_SIZE*(float)PATCH_SIZE);
		float g_val = (float)glSqSum2_val - powf(glSum2_val, 2) / ((float)PATCH_SIZE*(float)PATCH_SIZE);

		float correlation = ((float)p_val - q_val) / sqrtf(f_val * g_val);
		//printf(" \n Dla %d, %d   Uzyskana korrelacja = %f", tx, ty, correlation);
		correlate[tx + ty*block_side] = correlation;
	}

}