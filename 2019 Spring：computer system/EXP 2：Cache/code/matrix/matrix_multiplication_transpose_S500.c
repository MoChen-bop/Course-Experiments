#define SIZE 500


void product_with_transpose(int matrix_A[SIZE][SIZE], int matrix_B[SIZE][SIZE], int matrix_C[SIZE][SIZE]);
void transpose(int matrix[SIZE][SIZE]);


int main(int argc, char** argv)
{
	int matrix_A[SIZE][SIZE];
	int matrix_B[SIZE][SIZE];
	int matrix_C[SIZE][SIZE];

	transpose(matrix_B);

	product_with_transpose(matrix_A, matrix_B, matrix_C);

	return 0;
}


void transpose(int matrix[SIZE][SIZE])
{
	for (int i = 0; i < SIZE; i++) {
		for (int j = i; j < SIZE; j++) {
			int temp = matrix[i][j];
			matrix[i][j] = matrix[j][i];
			matrix[j][i] = temp;
		}
	}
}

void product_with_transpose(int matrix_A[SIZE][SIZE], int matrix_B[SIZE][SIZE], int matrix_C[SIZE][SIZE])
{
	for (int i = 0; i < SIZE; i++) {
		for (int j = 0; j < SIZE; j++) {
			int sum = 0;
			for (int k = 0; k < SIZE; k++) {
				sum += matrix_A[i][k] * matrix_B[j][k];
			}
			matrix_C[i][j] = sum;
		}
	}
}