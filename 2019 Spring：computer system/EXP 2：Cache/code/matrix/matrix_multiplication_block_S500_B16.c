#include <stdlib.h>

#define RANDOM_MAX 100
#define SIZE 500
#define BLOCK_SIZE 16

void fill_random_number(int matrix[SIZE][SIZE]);
void fill_zero(int matrix[SIZE][SIZE]);
void product_block(int matrix_A[SIZE][SIZE], int matrix_B[SIZE][SIZE], int matrix_C[SIZE][SIZE]);
void product_in_block(int A[SIZE][SIZE], int matrix_B[SIZE][SIZE], int matrix_C[SIZE][SIZE],
	int block_row, int block_column, int block_k);

int main(void)
{
	int matrix_A[SIZE][SIZE];
	int matrix_B[SIZE][SIZE];
	int matrix_C[SIZE][SIZE];
	// fill_random_number(matrix_A);
	// fill_random_number(matrix_B);
	// fill_zero(matrix_C);
	product_block(matrix_A, matrix_B, matrix_C);

	return 0;
}

void fill_random_number(int matrix[SIZE][SIZE])
{
	for (int i = 0; i < SIZE; i++)
		for (int j = 0; j < SIZE; j++)
			matrix[i][j] = rand() % RANDOM_MAX;
}

void fill_zero(int matrix[SIZE][SIZE])
{
	for (int i = 0; i < SIZE; i++)
		for (int j = 0; j < SIZE; j++)
			matrix[i][j] = 0;
}

void product_block(int A[SIZE][SIZE], int B[SIZE][SIZE], int C[SIZE][SIZE])
{
	for (int i = 0; i < SIZE; i+= BLOCK_SIZE)
		for (int j = 0 ; j < SIZE; j+= BLOCK_SIZE)
			for (int k = 0; k < SIZE; k += BLOCK_SIZE)
				product_in_block(A, B, C, i, j, k);
}

void product_in_block(int A[SIZE][SIZE], int B[SIZE][SIZE], int C[SIZE][SIZE], 
	int block_row, int block_column, int block_k)
{
	int end_row = (block_row + BLOCK_SIZE) > SIZE ? SIZE : (block_row + BLOCK_SIZE);
	int end_column = (block_column + BLOCK_SIZE) > SIZE ? SIZE : (block_column + BLOCK_SIZE);
	int end_k = (block_k + BLOCK_SIZE) > SIZE ? SIZE : (block_k + BLOCK_SIZE);
	for (int i = block_row; i < end_row; i++) 
		for (int j = block_column; j < end_column; j++) {
			int sum = 0;
			for (int k = block_k; k < end_k; k++)
				sum += A[i][k] * B[k][j];
			C[i][j] += sum;
		}

}