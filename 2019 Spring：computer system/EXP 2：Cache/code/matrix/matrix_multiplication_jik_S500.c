#include <stdlib.h>

#define RANDOM_MAX 100
#define SIZE 500

void fill_random_number(int matrix[SIZE][SIZE]);
void product_jik(int matrix_A[SIZE][SIZE], int matrix_B[SIZE][SIZE], int matrix_C[500][500]);

int main(void)
{
	int matrix_A[SIZE][SIZE];
	int matrix_B[SIZE][SIZE];
	int matrix_C[SIZE][SIZE];
	// fill_random_number(matrix_A);
	// fill_random_number(matrix_B);
	product_jik(matrix_A, matrix_B, matrix_C);

	return 0;
}

void fill_random_number(int matrix[SIZE][SIZE])
{
	for (int i = 0; i < SIZE; i++)
		for (int j = 0; j < SIZE; j++)
			matrix[i][j] = rand() % RANDOM_MAX;
}

void product_jik(int A[SIZE][SIZE], int B[SIZE][SIZE], int C[SIZE][SIZE])
{
	for (int j = 0; j < SIZE; j++)
		for (int i = 0; i < SIZE; i++) {
			int sum = 0;
			for (int k = 0; k < SIZE; k++) 
				sum += A[i][k] * B[k][j];
			C[i][j] = sum;
		}

}