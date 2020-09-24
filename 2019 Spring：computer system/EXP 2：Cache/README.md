# Experiment 2: Cache simulation

**Click Here!**
[计算机系统原理实验2](https://nbviewer.jupyter.org/github/MoChen-bop/Course-Experiments/blob/master/2019%20Spring%EF%BC%9Acomputer%20system/EXP%202%EF%BC%9ACache/计算机系统原理实验2.pdf)

## Objects

* To understand how cache works.
* To understand how internal settings affect cache's performance.
* To explore which way to access memory can improve program performance.
* Think about how to write a cache-friendly program.

## Details

Write a matrix multiplication program, then run it on gem5 simulator to get memory access trace. After that we need to write a cache simulator which inputs memory access trace, outputs several metrics to evaluate the performance of the memory access way of that program.

## Steps

#### Get trace

Write a program like this:

```c++
#define SIZE 500

int main(void)
{
	int matrix_A[SIZE][SIZE];
	int matrix_B[SIZE][SIZE];
	int matrix_C[SIZE][SIZE];
	product_ijk(matrix_A, matrix_B, matrix_C);

	return 0;
}

void product_ijk(int A[SIZE][SIZE], int B[SIZE][SIZE], int C[SIZE][SIZE])
{
	for (int i = 0; i < SIZE; i++)
		for (int j = 0; j < SIZE; j++) {
			int sum = 0;
			for (int k = 0; k < SIZE; k++) 
				sum += A[i][k] * B[k][j];
			C[i][j] = sum;
		}
}
```

Then run it in gem5 simulator to the trace. We can get when the CPU accessed memory and accessed which memory cell.

#### Write Cache simulator

I implemented three type cache:

* DMCache(Directed Mapping Cache)
* FAMCache(Fully Associated Mapping Cache)
* GAMCache(Group Associated Mapping Cache)

I run memory access trace that I get from step 1 in my own cache simulator with different replace strategy such as RR(random replace), LFUR(Least Frequently Used Replace), LRUR(Least Recently Used Replace) and FIFOR(First-In First-Out Replace).

#### Evaluate performance

There are several metrics to evaluate cache performance,

* **HC**: Hit Count, how many data that we want to access in cache.

* **MC**: Miss Count, how many times the data we want to access doesn't in cache.

* **AC**: Access Count, the total times CPU sends out memory access instruction, i.e. AC=HC+MC.

* **RC**: Read Count, count how many these memory access instructions are read-driven.

* **WC**: Write Count, count the total number of write-driven memory access instructions.

* **HR**: Hit Ratio, ![](http://latex.codecogs.com/gif.latex?\\HR=\frac{HC}{HC+MC})

* **MR**: Miss Ratio, , ![](http://latex.codecogs.com/gif.latex?\\MR=1-HR)

* **TRT**: Total Read Time, the time spent on read memory.

* **TWT**: Total Write Time, the time spent on write memory.

* **TAT**: Total Access Time, the time spent on access memory.

* **ART**: Average Read Time, ![](http://latex.codecogs.com/gif.latex?\\ART=\frac{TRT}{RC})

* **AWT**: Average Write Time, ![](http://latex.codecogs.com/gif.latex?\\AWT=\frac{TWT}{WC})

* **AAT**: Average Access Time, ![](http://latex.codecogs.com/gif.latex?\\AAT=\frac{TRT+TWT}{AC})

* **TP**: Through Put.

* **SUR**: Speed Up Ratio, ![](http://latex.codecogs.com/gif.latex?\\SUR=\frac{AC \times AML - TAT}{AC \times AML})

  Much more details are in file 实验数据.docx.

## Results

![](/img/results.png)







