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

__global__ void cudaSimpleGetBestCorrelate_v2(int* img2, int x, int y,
	int* im1_part, float med, float deviat, float* correlate){

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	const int block_side = SUB_IMG - PATCH_SIZE + 1;

	int thread_X = x + tx - block_side / 2;// centralny piksel analizowanego patcha
	int thread_Y = y + ty - block_side / 2;

	int sum = 0;
	float medium = 0.0;
	float deviation = 0.0;
	float numerator = 0.0;

	for (int row = 0; row < PATCH_SIZE; row++)
		for (int col = 0; col < PATCH_SIZE; col++)
		{
		int x_in_patch = thread_X + col - PATCH_SIZE / 2 ;//zero in 0,0, not x,y
		int y_in_patch = thread_Y + row - PATCH_SIZE / 2;
		//sum += shared_img2[x_start + (y_start)*SUB_IMG];
		sum += img2[x_in_patch + y_in_patch*IMAGE_WIDTH];
		}

		medium = float(sum) / ((float)PATCH_SIZE * (float)PATCH_SIZE);

		for (int row = 0; row < PATCH_SIZE; row++)
			for (int col = 0; col < PATCH_SIZE; col++)
			{
			int x_in_patch = thread_X + col - PATCH_SIZE / 2;//zero in 0,0, not x,y
			int y_in_patch = thread_Y + row - PATCH_SIZE / 2;
			int pixelIntensity = img2[x_in_patch + y_in_patch*IMAGE_WIDTH];

			deviation += pow((float)pixelIntensity - medium, 2);
			numerator += ((float)pixelIntensity - medium)*
				((float)im1_part[col + row*PATCH_SIZE] - med);
			}
		float correlation = numerator / sqrt(deviat*deviation);
		correlate[tx + ty*block_side] = correlation;
}

__global__ void cudaSimpleGetBestCorrelate(int* img2, int x, int y,
	int* im1_part, float med, float deviat, float* correlate){

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	const int block_side = SUB_IMG - PATCH_SIZE + 1;
	__shared__ int shared_img2[SUB_IMG*SUB_IMG];// copy all area to shared memory

	int thread_X = x + tx - SUB_IMG / 2;// uwzgledniamy i przesuniêcie centralnego piksela oraz caly patch dookola tego piksela
	int thread_Y = y + ty - SUB_IMG / 2;

	//zapisujemy caly obszar subImage do __shared__
	shared_img2[tx + SUB_IMG*ty] = img2[thread_Y*IMAGE_WIDTH + thread_X];
	__syncthreads();
	
	if (tx < block_side && ty < block_side)
	{
	int sum = 0;
	float medium = 0.0;
	float deviation = 0.0;
	float numerator = 0.0;

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

__global__ void cudaSimpleGetBestCorrelate_v3(int* img1, int* img2, int x, int y, float* max_correlate){

	int tx = threadIdx.x;
	int ty = threadIdx.y;

	const int block_side = SUB_IMG - PATCH_SIZE + 1;

	__shared__ int shared_img1[PATCH_SIZE * PATCH_SIZE];
	__shared__ int shared_img2[SUB_IMG*SUB_IMG];// copy all area to shared memory

	int thread_X = x + tx - SUB_IMG / 2;// uwzgledniamy i przesuniêcie centralnego piksela oraz caly patch dookola tego piksela
	int thread_Y = y + ty - SUB_IMG / 2;

	//zapisujemy caly obszar subImage do __shared__
	shared_img2[tx + SUB_IMG*ty] = img2[thread_Y*IMAGE_WIDTH + thread_X];
	
	if (tx < PATCH_SIZE && ty < PATCH_SIZE)//kopiujemy potrzebna czesc pierwszego obrazu do shared pamieci
		shared_img1[tx + PATCH_SIZE*ty] = img1[x + tx - PATCH_SIZE/2 + IMAGE_WIDTH*(y + ty - PATCH_SIZE / 2)];
	
	__syncthreads();

	if (tx < block_side && ty < block_side)
	{
		int sum1 = 0;
		int sum2 = 0;
		float medium1 = 0.0;
		float medium2 = 0.0;

		float deviation1 = 0.0;
		float deviation2 = 0.0;
		float numerator = 0.0;

		for (int row = 0; row < PATCH_SIZE; row++)
			for (int col = 0; col < PATCH_SIZE; col++)
			{
			int x_start = tx + col;//zero in 0,0, not x,y
			int y_start = ty + row;

			sum1 += shared_img1[col + (row)*PATCH_SIZE];
			sum2 += shared_img2[x_start + (y_start)*SUB_IMG];
			}

		medium1 = float(sum1) / ((float)PATCH_SIZE * (float)PATCH_SIZE);
		medium2 = float(sum2) / ((float)PATCH_SIZE * (float)PATCH_SIZE);

		for (int row = 0; row < PATCH_SIZE; row++)
			for (int col = 0; col < PATCH_SIZE; col++)
			{
			int x_start = tx + col;//zero in 0,0, not x,y
			int y_start = ty + row;
			deviation1 += pow(shared_img1[col + row*PATCH_SIZE] - medium1, 2);
			deviation2 += pow(shared_img2[x_start + y_start*SUB_IMG] - medium2, 2);
			

			numerator += ((float)shared_img1[col + row*PATCH_SIZE] - medium1)*
				((float)shared_img2[x_start + (y_start)*SUB_IMG] - medium2);
			}
		
		float correlation = numerator / sqrt(deviation1*deviation2);
		//printf("\n corr = %f", correlation);
		max_correlate[tx + ty*block_side] = correlation;
	}
	__syncthreads();

	
}

__global__ void cudaFastCorrelation(int* img1, int* img2, int x_init, int y_init, float* corr_mat){

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	//printf("\n sizeof(corr_mat) = %d\n", sizeof(corr_mat));
	//printf(" \n For [%d, %d] ", x_init - (SUB_IMG / 2 + 1) + bx * SUB_IMG + tx, y_init - (SUB_IMG / 2 + 1) + by * SUB_IMG + ty);
	
	if (bx * SUB_IMG + x_init + SUB_IMG / 2 < IMAGE_WIDTH &&
		by * SUB_IMG + y_init + SUB_IMG / 2 < IMAGE_HEIGHT)
	{
		//petla wykonywanego programu
		
		int x_ref = x_init + bx * SUB_IMG;
		int y_ref = y_init + by * SUB_IMG;

		__shared__ int shared_img1[PATCH_SIZE * PATCH_SIZE];
		__shared__ int shared_img2[SUB_IMG * SUB_IMG];// copy all area to shared memory



		if (tx < PATCH_SIZE && ty < PATCH_SIZE)//kopiujemy potrzebna czesc pierwszego obrazu do shared pamieci
			shared_img1[tx + PATCH_SIZE*ty] = img1[x_ref + tx - PATCH_SIZE / 2 + IMAGE_WIDTH*(y_ref + ty - PATCH_SIZE / 2)];


		int thread_X = x_ref + tx - SUB_IMG / 2;// uwzgledniamy i przesuniêcie centralnego piksela oraz caly patch dookola tego piksela
		int thread_Y = y_ref + ty - SUB_IMG / 2;

		//zapisujemy caly obszar subImage do __shared__
		shared_img2[tx + SUB_IMG*ty] = img2[thread_Y*IMAGE_WIDTH + thread_X];
		//printf(".");
		__syncthreads();

		const int block_side = SUB_IMG - PATCH_SIZE + 1;

		if (tx < block_side && ty < block_side)
		{
			int sum1 = 0;
			int sum2 = 0;
			float medium1 = 0.0;
			float medium2 = 0.0;

			float deviation1 = 0.0;
			float deviation2 = 0.0;
			float numerator = 0.0;

			for (int row = 0; row < PATCH_SIZE; row++)
				for (int col = 0; col < PATCH_SIZE; col++)
				{
				int x_start = tx + col;//zero in 0,0, not x,y
				int y_start = ty + row;

				sum1 += shared_img1[col + row * PATCH_SIZE];
				sum2 += shared_img2[x_start + y_start * SUB_IMG];
				}

			medium1 = float(sum1) / ((float)PATCH_SIZE * (float)PATCH_SIZE);
			medium2 = float(sum2) / ((float)PATCH_SIZE * (float)PATCH_SIZE);

			for (int row = 0; row < PATCH_SIZE; row++)
				for (int col = 0; col < PATCH_SIZE; col++)
				{
				int x_start = tx + col;//zero in 0,0, not x,y
				int y_start = ty + row;
				deviation1 += pow((float)shared_img1[col     + row*PATCH_SIZE ] - medium1, 2);
				deviation2 += pow((float)shared_img2[x_start + y_start*SUB_IMG] - medium2, 2);

				numerator += ((float)shared_img1[col + row*PATCH_SIZE] - medium1)*
					((float)shared_img2[x_start + (y_start)*SUB_IMG] - medium2);
				}

			float correlation = numerator / sqrt(deviation1*deviation2);

			int x_corr = x_ref - (SUB_IMG / 2 + 1);
			int y_corr = y_ref - (SUB_IMG / 2 + 1);
			int block_position = tx + ty*block_side;
			int glWidth = IMAGE_WIDTH - SUB_IMG + 1;

			int index = block_position + block_side*block_side*(
				x_corr + y_corr*glWidth);


			corr_mat[index] = correlation;
			//if (x_init == 9 && y_init == 10 && bx == 0 && by == 0 && tx == 0 && ty == 0)
				//printf(" \n For x_init = %d y_init = %d [%d, %d] corr before = %f", x_init, y_init, x_corr + tx, y_corr + ty, corr_mat[index]);

			//if (x_ref == 24 && y_ref == 23 && x_corr + tx == 21 && y_corr + ty == 21)
			//	printf(" \n\n For [%d, %d] and [%d, %d] corr before = %f \n ", x_ref, y_ref, x_corr + tx, y_corr + ty, corr_mat[index]);
			__syncthreads();
		}
		__syncthreads();
	}

}

__global__ void  cudaFastCorrelation(float* corrMat_dev, float* final_corrMat){
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int x = tx;
	int y = ty;

	int glWidth = IMAGE_WIDTH - SUB_IMG + 1;
	int glHeight = IMAGE_HEIGHT - SUB_IMG + 1;
	const int block_side = SUB_IMG - PATCH_SIZE + 1;
	const int zero_index = block_side*block_side*(x + y * glWidth);

	float max_corr = -1.0;
	// pozniej mozna zamienic na strukture piecioelementowa.
	for (int i = 0; i < block_side * block_side; i++)
		max_corr = max_corr > corrMat_dev[zero_index + i] ? max_corr : corrMat_dev[zero_index + i];

	__syncthreads();

	printf("\n max corr = %f dla indexow = %d i %d ", max_corr, x, y);
	final_corrMat[x + y*glWidth] = max_corr;
}

__global__ void cudaFastCorrelation3DMatrix(int* img1, int* img2, int x_init, int y_init, float** corr_mat){

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	if (bx * SUB_IMG + SUB_IMG / 2 < IMAGE_WIDTH &&
		by * SUB_IMG + SUB_IMG / 2 < IMAGE_HEIGHT)
	{
		//petla wykonywanego programu

		int x_ref = x_init + bx * SUB_IMG;
		int y_ref = y_init + by * SUB_IMG;

		__shared__ int shared_img1[PATCH_SIZE * PATCH_SIZE];
		__shared__ int shared_img2[SUB_IMG * SUB_IMG];// copy all area to shared memory

		int thread_X = x_ref + tx - SUB_IMG / 2;// uwzgledniamy i przesuniêcie centralnego piksela oraz caly patch dookola tego piksela
		int thread_Y = y_ref + ty - SUB_IMG / 2;

		//zapisujemy caly obszar subImage do __shared__
		shared_img2[tx + SUB_IMG*ty] = img2[thread_Y*IMAGE_WIDTH + thread_X];

		if (tx < PATCH_SIZE && ty < PATCH_SIZE)//kopiujemy potrzebna czesc pierwszego obrazu do shared pamieci
			shared_img1[tx + PATCH_SIZE*ty] = img1[x_ref + tx - PATCH_SIZE / 2 + IMAGE_WIDTH*(y_ref + ty - PATCH_SIZE / 2)];

		__syncthreads();

		const int block_side = SUB_IMG - PATCH_SIZE + 1;

		if (tx < block_side && ty < block_side)
		{
			int sum1 = 0;
			int sum2 = 0;
			float medium1 = 0.0;
			float medium2 = 0.0;

			float deviation1 = 0.0;
			float deviation2 = 0.0;
			float numerator = 0.0;

			for (int row = 0; row < PATCH_SIZE; row++)
				for (int col = 0; col < PATCH_SIZE; col++)
				{
				int x_start = tx + col;//zero in 0,0, not x,y
				int y_start = ty + row;

				sum1 += shared_img1[col + (row)*PATCH_SIZE];
				sum2 += shared_img2[x_start + (y_start)*SUB_IMG];
				}

			medium1 = float(sum1) / ((float)PATCH_SIZE * (float)PATCH_SIZE);
			medium2 = float(sum2) / ((float)PATCH_SIZE * (float)PATCH_SIZE);

			for (int row = 0; row < PATCH_SIZE; row++)
				for (int col = 0; col < PATCH_SIZE; col++)
				{
				int x_start = tx + col;//zero in 0,0, not x,y
				int y_start = ty + row;
				deviation1 += pow(shared_img1[col + row*PATCH_SIZE] - medium1, 2);
				deviation2 += pow(shared_img2[x_start + y_start*SUB_IMG] - medium2, 2);

				numerator += ((float)shared_img1[col + row*PATCH_SIZE] - medium1)*
					((float)shared_img2[x_start + (y_start)*SUB_IMG] - medium2);
				}

			float correlation = numerator / sqrt(deviation1*deviation2);


			int x_corr = x_init - (SUB_IMG / 2 + 1) + bx * SUB_IMG;
			int y_corr = y_init - (SUB_IMG / 2 + 1) + by * SUB_IMG;
			corr_mat[x_corr + y_corr * (IMAGE_WIDTH - (SUB_IMG - 1))][tx + ty*block_side] = correlation;
		}
	}

}