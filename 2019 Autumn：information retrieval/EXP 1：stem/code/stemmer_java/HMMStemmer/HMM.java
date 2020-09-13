package HMMStemmer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.logging.Logger;

public class HMM {
	private double precision = 1e-7;

	private int sequenceLen;

	protected double[] pi;
	protected double[][] transferProbability;
	protected double[][] emissionProbability;

	protected double[][] accumulated_gamma;
	protected double[][] accumulated_ksi;
	protected double[] accumulated_pi;
	
	protected int accumulated_count;

	protected int stateNum;
	protected int observationNum;

	public HMM() {
		super();
	}

	public HMM(int stateNum, int observationNum, double[] pi, double[][] transferProbability,
			double[][] emissionProbability) {
		this.stateNum = stateNum;
		this.observationNum = observationNum;
		this.pi = pi;
		this.transferProbability = transferProbability;
		this.emissionProbability = emissionProbability;
		init();
	}

	public HMM(int stateNum, int observationNum) {
		this.stateNum = stateNum;
		this.observationNum = observationNum;
		initParameters();
	}

	public void train(int[] x, int maxIter, double precision) {
		this.sequenceLen = x.length;
		baumWelch(x, maxIter, precision);
	}

	public void train(int[] x) {
		this.sequenceLen = x.length;
		baumWelch(x);
	}

	protected void baumWelch(int[] x, int maxIter, double precision) {
		int iter = 0;
		double oldMaxError = 0;
		if (maxIter <= 0) {
			maxIter = Integer.MAX_VALUE;
		}

		this.sequenceLen = x.length;
		double[][] alpha = new double[sequenceLen][stateNum];
		double[][] beta = new double[sequenceLen][stateNum];
		double[][] gamma = new double[sequenceLen][stateNum];
		double[][][] ksi = new double[sequenceLen][stateNum][stateNum];

		while (iter < maxIter) {

			calcAlpha(x, alpha);
			calcBeta(x, beta);
			calcGamma(x, alpha, beta, gamma);
			calcKsi(x, alpha, beta, ksi);

			double[][] oldA = generateOldA();
			updateLambda(x, gamma, ksi);

			double maxError = calcError(oldA, null, null);
			if (maxError < precision || (Math.abs(maxError - oldMaxError)) < this.precision) {
				break;
			}

			oldMaxError = maxError;
			iter++;
		}
	}

	protected void baumWelch(int[] x) {

		this.sequenceLen = x.length;
		double[][] alpha = new double[sequenceLen][stateNum];
		double[][] beta = new double[sequenceLen][stateNum];
		double[][] gamma = new double[sequenceLen][stateNum];
		double[][][] ksi = new double[sequenceLen][stateNum][stateNum];

		calcAlpha(x, alpha);
		calcBeta(x, beta);
		calcGamma(x, alpha, beta, gamma);
		calcKsi(x, alpha, beta, ksi);

		accumulateProbability(x, gamma, ksi);
	}

	protected double[][] generateOldA() {
		double[][] oldA = new double[stateNum][stateNum];
		for (int i = 0; i < stateNum; i++) {
			for (int j = 0; j < stateNum; j++) {
				oldA[i][j] = transferProbability[i][j];
			}
		}

		return oldA;
	}

	protected double[][] generateOldB() {
		double[][] oldB = new double[stateNum][observationNum];
		for (int i = 0; i < stateNum; i++) {
			for (int j = 0; j < observationNum; j++) {
				oldB[i][j] = emissionProbability[i][j];
			}
		}
		return oldB;
	}

	protected double calcError(double[][] oldA, double[] oldPi, double[][] oldB) {
		double maxError = 0;
		for (int i = 0; i < stateNum; i++) {
			for (int j = 0; j < stateNum; j++) {
				double tmp = Math.abs(oldA[i][j] - transferProbability[i][j]);
				maxError = tmp > maxError ? tmp : maxError;
			}
		}

		return maxError;
	}

	public void initParameters() {
		pi = new double[stateNum];
		transferProbability = new double[stateNum][stateNum];
		emissionProbability = new double[stateNum][observationNum];

		for (int i = 0; i < stateNum; i++) {
			pi[i] = 0;

			for (int j = 0; j < stateNum; j++) {
				transferProbability[i][j] = 0;
			}

			for (int k = 0; k < observationNum; k++) {
				emissionProbability[i][k] = 0;
			}
		}

		init();
	}

	public void init() {
		this.accumulated_gamma = new double[this.observationNum][this.stateNum];
		this.accumulated_ksi = new double[this.stateNum][this.stateNum];
		this.accumulated_pi = new double[this.stateNum];
		this.accumulated_count = 0;

		clear_accumulated();
	}
	
	public void clear_accumulated() {
		this.accumulated_count = 0;
		for (int i = 0; i < accumulated_gamma.length; i++) {
			for (int j = 0; j < accumulated_gamma[0].length; j++) {
				accumulated_gamma[i][j] = 0;
			}
		}

		for (int i = 0; i < accumulated_ksi.length; i++) {
			for (int j = 0; j < accumulated_ksi[0].length; j++) {
				accumulated_ksi[i][j] = 0;
			}
		}
		
		for (int i = 0; i < accumulated_pi.length; i++) {
			accumulated_pi[i] = 0;
		}
	}

	public void clear(double[][] array) {
		for (int i = 0; i < array.length; i++) {
			for (int j = 0; j < array[0].length; j++) {
				array[i][j] = 0;
			}
		}
	}

	protected void calcAlpha(int[] x, double[][] alpha) {
		for (int i = 0; i < stateNum; i++) {
			alpha[0][i] = pi[i] * emissionProbability[i][x[0]];
		}

		double[] probaArr = new double[stateNum];
		for (int t = 1; t < x.length; t++) {
			for (int i = 0; i < stateNum; i++) {
				for (int j = 0; j < stateNum; j++) {
					probaArr[j] = (alpha[t - 1][j] * transferProbability[j][i]);
				}
				alpha[t][i] = sum(probaArr) * emissionProbability[i][x[t]];
			}
		}
	}

	protected void calcBeta(int[] x, double[][] beta) {
		for (int i = 8; i < stateNum; i++) {
			beta[x.length - 1][i] = 1;
		}

		double[] probaArr = new double[stateNum];
		for (int t = x.length - 2; t >= 0; t--) {
			for (int i = 0; i < stateNum; i++) {
				for (int j = 0; j < stateNum; j++) {
					probaArr[j] = transferProbability[i][j] * emissionProbability[j][x[t + 1]] * beta[t + 1][j];
				}
				beta[t][i] = sum(probaArr);
			}
		}
	}

	protected void calcKsi(int[] x, double[][] alpha, double[][] beta, double[][][] ksi) {
		double[] probaArr = new double[stateNum * stateNum];
		for (int t = 0; t < sequenceLen - 1; t++) {
			int k = 0;
			for (int i = 0; i < stateNum; i++) {
				for (int j = 0; j < stateNum; j++) {
					ksi[t][i][j] = alpha[t][i] * transferProbability[i][j] * emissionProbability[j][x[t + 1]]
							* beta[t + 1][j];
					probaArr[k++] = ksi[t][i][j];
				}
			}

			double sumProb = sum(probaArr);
			for (int i = 0; i < stateNum; i++) {
				for (int j = 0; j < stateNum; j++) {
					ksi[t][i][j] /= sumProb;
				}
			}

		}
	}

	protected void calcGamma(int[] x, double[][] alpha, double[][] beta, double[][] gamma) {
		for (int t = 0; t < x.length; t++) {
			for (int i = 0; i < stateNum; i++) {
				gamma[t][i] = alpha[t][i] * beta[t][i];
			}

			double sumProb = sum(gamma[t]);
			for (int j = 0; j < stateNum; j++) {
				gamma[t][j] = gamma[t][j] / sumProb;
			}
		}

	}

	protected void accumulateProbability(int[] x, double[][] gamma, double[][][] ksi) {
		this.accumulated_count++;
		
		for (int i = 0; i < this.stateNum; i++) {
			this.accumulated_pi[i] += gamma[0][i];
		}
		
		int T = x.length;
		for (int t = 0; t < T; t++) {
			for (int j = 0; j < this.stateNum; j++) {
				this.accumulated_gamma[x[t]][j] += gamma[t][j];
			}
		}

		for (int i = 0; i < this.stateNum; i++) {
			for (int j = 0; j < this.stateNum; j++) {
				for (int t = 0; t < T - 1; t++) {
					accumulated_ksi[i][j] += ksi[t][i][j];
				}
			}
		}
	}

	public double modifyLambda() {
		double[][] oldA = generateOldA();
		
		modifyPi();
		modifyA();
		modifyB();
		
		clear_accumulated();
		double maxError = calcError(oldA, null, null);
		
		return maxError;		
	}

	public void modifyPi() {
		for (int i = 0; i < this.stateNum; i++) {
			this.pi[i] = this.accumulated_pi[i] / this.accumulated_count;
		}
	}

	public void modifyA() {
		for (int i = 0; i < this.stateNum; i++) {
			double sum = 0;
			for (int k = 0; k < this.observationNum; k++) {
				sum += this.accumulated_gamma[k][i];
			}
			for (int j = 0; j < this.stateNum; j++) {
				if (sum == 0) 
					this.transferProbability[i][j] = 0;
				else
					this.transferProbability[i][j] = this.accumulated_ksi[i][j] / sum;
			}
		}
	}

	public void modifyB() {
		for (int i = 0; i < this.stateNum; i++) {
			double sum = 0;
			for (int k = 0; k < this.observationNum; k++) {
				sum += this.accumulated_gamma[k][i];
			}
			
			for (int j = 0; j < this.observationNum; j++) {
				if (sum == 0) 
					this.emissionProbability[i][j] = 0;
				else
					this.emissionProbability[i][j] = this.accumulated_gamma[j][i] / sum;
			}
		}
	}

	protected void updateLambda(int[] x, double[][] gamma, double[][][] ksi) {
		updatePi(gamma);
		updateA(ksi, gamma);
		updateB(x, gamma);
	}

	public void updatePi(double[][] gamma) {
		for (int i = 0; i < stateNum; i++) {
			pi[i] = gamma[0][i];
		}
	}

	protected void updateA(double[][][] ksi, double[][] gamma) {
		double[] gammaSum = new double[stateNum];
		double[] tmp = new double[sequenceLen - 1];
		for (int i = 0; i < stateNum; i++) {
			for (int t = 0; t < sequenceLen - 1; t++) {
				tmp[t] = gamma[t][i];
			}
			gammaSum[i] = sum(tmp);
		}
		double[] ksiProbArr = new double[sequenceLen - 1];
		for (int i = 0; i < stateNum; i++) {
			for (int j = 0; j < stateNum; j++) {
				for (int t = 0; t < sequenceLen - 1; t++) {
					ksiProbArr[t] = ksi[t][i][j];
				}
				if (gammaSum[i] == 0)
					transferProbability[i][j] = 0;
				else
					transferProbability[i][j] = sum(ksiProbArr) / gammaSum[i];
			}
		}
	}

	protected void updateB(int[] x, double[][] gamma) {
		double[] gammaSum2 = new double[stateNum];
		double[] tmp2 = new double[sequenceLen];
		for (int i = 0; i < stateNum; i++) {
			for (int t = 0; t < sequenceLen; t++) {
				tmp2[t] = gamma[t][i];
			}
			gammaSum2[i] = sum(tmp2);
		}

		ArrayList<Double> valid = new ArrayList<Double>();
		for (int i = 0; i < stateNum; i++) {
			for (int k = 0; k < observationNum; k++) {
				valid.clear();
				for (int t = 0; t < sequenceLen; t++) {
					if (x[t] == k) {
						valid.add(gamma[t][i]);
					}
				}

				if (valid.size() == 0) {
					emissionProbability[i][k] = 0;
					continue;
				}

				double[] validArr = new double[valid.size()];
				for (int q = 0; q < valid.size(); q++) {
					validArr[q] = valid.get(q);
				}

				double validSum = sum(validArr);
				emissionProbability[i][k] = validSum / gammaSum2[i];

				// if (gammaSum2[i] == 0)
				// emissionProbability[i][k] = 1;
				// else
				// emissionProbability[i][k] = validSum / gammaSum2[i];
			}
		}
	}

	public double sum(double[] probaArr) {
		if (probaArr.length == 0) {
			return 0;
		}

		double result = 0;
		for (int i = 0; i < probaArr.length; i++) {
			result += probaArr[i];
		}
		return result;
	}

	public double max(double[] logProbaArr) {
		double m = logProbaArr[0];
		for (int i = 1; i < logProbaArr.length; i++) {
			if (m < logProbaArr[i]) {
				m = logProbaArr[i];
			}
		}
		return m;
	}

	public int[] verterbi(int[] o) {
		double[][] deltas = new double[o.length][this.stateNum];
		int[][] states = new int[o.length][this.stateNum];
		for (int i = 0; i < this.stateNum; i++) {
			deltas[0][i] = pi[i] * emissionProbability[i][o[0]];
		}

		for (int t = 1; t < o.length; t++) {
			for (int i = 0; i < this.stateNum; i++) {
				deltas[t][i] = deltas[t - 1][0] * transferProbability[0][i];
				for (int j = 1; j < this.stateNum; j++) {
					double tmp = deltas[t - 1][j] * transferProbability[j][i];
					if (tmp > deltas[t][i]) {
						deltas[t][i] = tmp;
						states[t][i] = j;
					}
				}
				deltas[t][i] *= emissionProbability[i][o[t]];
			}
		}

		int[] predict = new int[o.length];
		double max = deltas[o.length - 1][0];
		for (int i = 1; i < this.stateNum; i++) {
			if (deltas[o.length - 1][i] > max) {
				max = deltas[o.length - 1][i];
				predict[o.length - 1] = i;
			}
		}

		for (int i = o.length - 2; i >= 0; i--) {
			predict[i] = states[i + 1][predict[i + 1]];
		}

		return predict;
	}

	public int[] mostLikely(int[] o) {
		double[][] alpha = new double[o.length][this.stateNum];
		double[][] beta = new double[o.length][this.stateNum];
		double[][] gamma = new double[o.length][this.stateNum];
		int[] predict = new int[o.length];

		this.calcAlpha(o, alpha);
		this.calcBeta(o, beta);
		this.calcGamma(o, alpha, beta, gamma);

		for (int i = 0; i < o.length; i++) {
			double max = gamma[i][0];
			int max_index = 0;
			for (int j = 1; j < this.stateNum; j++) {
				if (gamma[i][j] > max) {
					max = gamma[i][j];
					max_index = j;
				}
			}
			predict[i] = max_index;
		}

		return predict;
	}

	public void printInfo() {
		System.out.println("Pi: ");
		for (int i = 0; i < pi.length; i++) {
			System.out.print(pi[i] + " ");
		}
		System.out.println();
		System.out.println("TransferProbability: ");
		for (int i = 0; i < transferProbability.length; i++) {
			for (int j = 0; j < transferProbability[0].length; j++) {
				System.out.print(transferProbability[i][j] + " ");
			}
			System.out.println();
		}
		System.out.println();
		System.out.println("EmissionProbability: ");
		for (int i = 0; i < emissionProbability.length; i++) {
			for (int j = 0; j < emissionProbability[0].length; j++) {
				System.out.print(emissionProbability[i][j] + " ");
			}
			System.out.println();
		}

	}
}
