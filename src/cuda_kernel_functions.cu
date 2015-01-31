#include "cuda_kernel_functions.cuh"
#include <math.h>


__global__ void cudaFastCorrelation(int* img1, int* img2, int x_init, int y_init, float* corr_mat){

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	//printf("\n sizeof(corr_mat) = %d\n", sizeof(corr_mat));
	//printf(" \n For [%d, %d] ", x_init - (SUB_IMG / 2 + 1) + bx * SUB_IMG + tx, y_init - (SUB_IMG / 2 + 1) + by * SUB_IMG + ty);
	
	if (bx * SUB_IMG + x_init + SUB_IMG / 2 <= IMAGE_WIDTH&&
		by * SUB_IMG + y_init + SUB_IMG / 2 <= IMAGE_HEIGHT)
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

			int x_corr = x_ref - (SUB_IMG / 2 + 1);
			int y_corr = y_ref - (SUB_IMG / 2 + 1);
			int block_position = tx + ty*block_side;
			int glWidth = IMAGE_WIDTH - SUB_IMG + 1;

			int index = block_position + block_side*block_side*(
				x_corr + y_corr*glWidth);

			//float correlation = numerator / sqrt(deviation1*deviation2);
			//corr_mat[index] = correlation;
			
			
			if (numerator == 0 )
				corr_mat[index] = 1;
			else
			{
				float correlation = numerator / sqrt(deviation1*deviation2);
				corr_mat[index] = correlation;
			}
				//printf("\n corr = %f ", correlation);

			//__syncthreads();
		}
		//__syncthreads();
	}

}



__global__ void  cudaGetMaxValues(float* corrMat_dev, float* final_corrMat, int* posMat){
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int x = bx * 32 + tx;
	int y = by * 32 + ty;
	int glWidth = IMAGE_WIDTH - SUB_IMG + 1;
	int glHeight = IMAGE_HEIGHT - SUB_IMG + 1;

	if (x < glWidth && y < glHeight)
	{
		const int block_side = SUB_IMG - PATCH_SIZE + 1;
		const int zero_index = block_side*block_side*(x + y * glWidth);

		float max_corr = -1.0;
		int position = 0;

		int middle = (block_side / 2 + 1)*(block_side / 2 + 1);
		if (corrMat_dev[zero_index + middle] == 1)
		{
			final_corrMat[x + y*glWidth] = 1.0;
			posMat[x + y*glWidth] = middle;
		}
		else{
			for (int i = 0; i < block_side * block_side; i++)
			{
				float correlation = corrMat_dev[zero_index + i];
				if (correlation > max_corr)
				{
					max_corr = correlation;
					position = i;
				}
			}
			final_corrMat[x + y*glWidth] = max_corr;
			posMat[x + y*glWidth] = position;
		}
	}
}