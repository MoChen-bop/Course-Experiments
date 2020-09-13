#define SIZE 500
#define BLOCK_SIZE 16

void product_with_transpose(int matrix_A[SIZE][SIZE], int matrix_B[SIZE][SIZE], int matrix_C[SIZE][SIZE]);
void transpose_with_block(int matrix[SIZE][SIZE]);


int main()
{
	int matrix_A[SIZE][SIZE];
	int matrix_B[SIZE][SIZE];
	int matrix_C[SIZE][SIZE];

	transpose_with_block(matrix_B);
	product_with_transpose(matrix_A, matrix_B, matrix_C);

	return 0;
}

void transpose_with_block(int matrix[SIZE][SIZE])
{
	for (int i_b = 0; i_b < SIZE; i_b += BLOCK_SIZE) {
		for (int j_b = i_b; j_b < SIZE; j_b += BLOCK_SIZE) {
			int i_end = (i_b + BLOCK_SIZE) > SIZE ? SIZE : (i_b + BLOCK_SIZE);
			int j_end = (j_b + BLOCK_SIZE) > SIZE ? SIZE : (j_b + BLOCK_SIZE);
			if (i_b == j_b) {
				for (int i = i_b; i < i_end; i++)
					for (int j = j_b + (i - i_b); j < j_end; j++) {
						int temp = matrix[i][j];
						matrix[i][j] = matrix[j][i];
						matrix[j][i] = temp;
					}
			}
			else {
				for (int i = i_b; i < i_end; i++)
					for (int j = j_b; j < j_end; j++) {
						int temp = matrix[i][j];
						matrix[i][j] = matrix[j][i];
						matrix[j][i] = temp;
					}
			}
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
