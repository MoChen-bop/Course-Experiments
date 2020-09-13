package HMM;

import java.util.Arrays;

public class testHMM {
	public static void main(String[] args) {
		int[] x1 = {0, 1, 1, 0, 2};
		int[] x2 = {1, 0, 0, 1, 1, 2, 2};
		int[] x3 = {1, 0, 2, 2};
		double[] pi = {0, (double) -Math.pow(2, 31), 
				(double) -Math.pow(2, 31), (double) -Math.pow(2, 31)};
		double[][] transferProbability1 = {
				{0, 0, (double) -Math.pow(2, 31), (double) -Math.pow(2, 31)},
				{(double) -Math.pow(2, 31), (double) -Math.pow(2, 31), 0, (double) -Math.pow(2, 31)},
				{(double) -Math.pow(2, 31), (double) -Math.pow(2, 31), (double) -Math.pow(2, 31), 0},
				{(double) -Math.pow(2, 31), (double) -Math.pow(2, 31), (double) -Math.pow(2, 31), (double) -Math.pow(2, 31)},
		};
		double[][] emissionProbability =  {
				{0, 0, (double) -Math.pow(2, 31)},
				{0, 0, (double) -Math.pow(2, 31)},
				{0, 0, 0},
				{0, 0, 0},
		};
		
		UnsupervisedFirstOrderGeneralHMM model = new UnsupervisedFirstOrderGeneralHMM(4, 3, 
				pi, transferProbability1, emissionProbability);
		model.train(x1, 1, 0.001);
		int[] x4 = model.verterbi(x3);
		System.out.println(Arrays.toString(x4));
		model.train(x2, 1, 0.001);
		x4 = model.verterbi(x3);
		System.out.println(Arrays.toString(x4));
	}

}
