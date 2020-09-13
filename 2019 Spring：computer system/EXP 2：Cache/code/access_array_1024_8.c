#define SIZE 1024
#define STRIDE 8
#define LOOP 1000

int array[SIZE];

int main(void)
{
	int a;
	int index = 0;
	for (int i = 0; i < LOOP; i++) {
		a += array[index];
		index = (index + STRIDE) % SIZE;
	}
	return 0;
}