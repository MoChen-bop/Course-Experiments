#include <stdlib.h>

#define RANDOM_MAX 100
#define SIZE 500

void fill_random_number(int matrix[SIZE][SIZE]);
void fill_zero(int matrix[SIZE][SIZE]);
void product_jki(int matrix_A[SIZE][SIZE], int matrix_B[SIZE][SIZE], int matrix_C[SIZE][SIZE]);

int main(void)
{
	int matrix_A[SIZE][SIZE];
	int matrix_B[SIZE][SIZE];
	int matrix_C[SIZE][SIZE];
	// fill_random_number(matrix_A);
	// fill_random_number(matrix_B);
	// fill_zero(matrix_C);
	product_jki(matrix_A, matrix_B, matrix_C);

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

void product_jki(int A[SIZE][SIZE], int B[SIZE][SIZE], int C[SIZE][SIZE])
{
	for (int j = 0; j < SIZE; j++)
		for (int k = 0; k < SIZE; k++) {
			int temp = B[k][j];
			for (int i = 0; i < SIZE; i++) 
				C[i][j] += A[i][k] * temp;
		}
}