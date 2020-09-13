package HMM;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.logging.Logger;

public class UnsupervisedFirstOrderGeneralHMM {
	private double precision = 1e-7;

	private int sequenceLen;
	public Logger logger = Logger.getLogger(UnsupervisedFirstOrderGeneralHMM.class.getName());

	protected double[] pi;
	protected double[][] transferProbability1;
	protected double[][] emissionProbability;

	public static final double INFINITY = (double) -Math.pow(2, 31);
	protected int stateNum;
	protected int observationNum;

	public UnsupervisedFirstOrderGeneralHMM() {
		super();
	}

	public UnsupervisedFirstOrderGeneralHMM(int stateNum, int observationNum, double[] pi,
			double[][] transferProbability1, double[][] emissionProbability) {
		this.stateNum = stateNum;
		this.observationNum = observationNum;
		this.pi = pi;
		this.transferProbability1 = transferProbability1;
		this.emissionProbability = emissionProbability;
	}

	public UnsupervisedFirstOrderGeneralHMM(int stateNum, int observationNum) {
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
	}

	protected void baumWelch(int[] x, int maxIter, double precision) {
		int iter = 0;
		double oldMaxError = 0;
		if (maxIter <= 0) {
			maxIter = Integer.MAX_VALUE;
		}

		double[][] alpha = new double[sequenceLen][stateNum];
		double[][] beta = new double[sequenceLen][stateNum];
		double[][] gamma = new double[sequenceLen][stateNum];
		double[][][] ksi = new double[sequenceLen][stateNum][stateNum];

		while (iter < maxIter) {
			logger.info("\niter" + iter + "...");
			long start = System.currentTimeMillis();

			calcAlpha(x, alpha);
			calcBeta(x, beta);
			calcGamma(x, alpha, beta, gamma);
			calcKsi(x, alpha, beta, ksi);

			double[][] oldA = generateOldA();
			updateLambda(x, gamma, ksi);

			double maxError = calcError(oldA, null, null);
			logger.info("max_error: " + maxError);
			if (maxError < precision || (Math.abs(maxError - oldMaxError)) < this.precision) {
				logger.info("parameters has converged...");
				break;
			}

			oldMaxError = maxError;
			iter++;
			long end = System.currentTimeMillis();
			logger.info("iteration finished, time comsumed: " + (end - start) + "ms");

			logger.info("pi: " + Arrays.toString(pi));
			logger.info("A: ");
			for (int i = 0; i < transferProbability1.length; i++) {
				logger.info(Arrays.toString(transferProbability1[i]));
			}

		}
		logger.info("final parameters: ");
		logger.info("pi: " + Arrays.toString(pi));
		for (int i = 0; i < pi.length; i++) {
			System.out.print(Math.exp(pi[i]) + " ");
		}
		System.out.println();
		
		logger.info("transferProbability: ");
		for (int i = 0; i < transferProbability1.length; i++) {
			logger.info(Arrays.toString(transferProbability1[i]));
			for (int j = 0; j < transferProbability1[0].length; j++) {
				System.out.print(Math.exp(transferProbability1[i][j]) + " ");
			}
			System.out.println();
		}
		
		logger.info("emissionProbability: ");
		for (int i = 0; i < emissionProbability.length; i++) {
			logger.info(Arrays.toString(emissionProbability[i]));
			for (int j = 0; j < emissionProbability[0].length; j++) {
				System.out.print(Math.exp(emissionProbability[i][j]) + " ");
			}
			System.out.println();
		}
	}

	protected double[][] generateOldA() {
		double[][] oldA = new double[stateNum][stateNum];
		for (int i = 0; i < stateNum; i++) {
			for (int j = 0; j < stateNum; j++) {
				oldA[i][j] = transferProbability1[i][j];
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
				double tmp = Math.abs(oldA[i][j] - transferProbability1[i][j]);
				maxError = tmp > maxError ? tmp : maxError;
			}
		}

		return maxError;
	}

	public void initParameters() {
		pi = new double[stateNum];
		transferProbability1 = new double[stateNum][stateNum];
		emissionProbability = new double[stateNum][observationNum];

		for (int i = 0; i < stateNum; i++) {
			pi[i] = INFINITY;

			for (int j = 0; j < stateNum; j++) {
				transferProbability1[i][j] = INFINITY;
			}

			for (int k = 0; k < observationNum; k++) {
				emissionProbability[i][k] = INFINITY;
			}
		}
	}

	protected void calcAlpha(int[] x, double[][] alpha) {
		logger.info("calculate alpha...");
		long start = System.currentTimeMillis();

		for (int i = 0; i < stateNum; i++) {
			alpha[0][i] = pi[i] + emissionProbability[i][x[0]];
		}

		double[] logProbaArr = new double[stateNum];
		for (int t = 1; t < sequenceLen; t++) {
			for (int i = 0; i < stateNum; i++) {
				for (int j = 0; j < stateNum; j++) {
					logProbaArr[j] = (alpha[t - 1][j] + transferProbability1[j][i]);
				}
				alpha[t][i] = logSum(logProbaArr) + emissionProbability[i][x[t]];
			}
		}

		long end = System.currentTimeMillis();
		logger.info("calculate finished..., time comsumed: " + (end + start) + "ms");

	}

	protected void calcBeta(int[] x, double[][] beta) {
		logger.info("calculate beta...");
		long start = System.currentTimeMillis();

		for (int i = 0; i < stateNum; i++) {
			beta[sequenceLen - 1][i] = 1;
		}

		double[] logProbaArr = new double[stateNum];
		for (int t = sequenceLen - 2; t >= 0; t--) {
			for (int i = 0; i < stateNum; i++) {
				for (int j = 0; j < stateNum; j++) {
					logProbaArr[j] = transferProbability1[i][j] + emissionProbability[j][x[t + 1]] + beta[t + 1][j];
				}
				beta[t][i] = logSum(logProbaArr);
			}
		}

		long end = System.currentTimeMillis();
		logger.info("calculate finished..., time comsumed: " + (end - start) + "ms");

	}

	protected void calcKsi(int[] x, double[][] alpha, double[][] beta, double[][][] ksi) {
		logger.info("calculate ksi...");
		long start = System.currentTimeMillis();

		double[] logProbaArr = new double[stateNum * stateNum];
		for (int t = 0; t < sequenceLen - 1; t++) {
			int k = 0;
			for (int i = 0; i < stateNum; i++) {
				for (int j = 0; j < stateNum; j++) {
					ksi[t][i][j] = alpha[t][i] + transferProbability1[i][j] + emissionProbability[j][x[t + 1]]
							+ beta[t + 1][j];
					logProbaArr[k++] = ksi[t][i][j];
				}
			}

			double logSum = logSum(logProbaArr);
			for (int i = 0; i < stateNum; i++) {
				for (int j = 0; j < stateNum; j++) {
					ksi[t][i][j] -= logSum;
				}
			}

			long end = System.currentTimeMillis();
			logger.info("calculate finished: " + (end - start) + "ms");

		}
	}

	protected void calcGamma(int[] x, double[][] alpha, double[][] beta, double[][] gamma) {
		logger.info("calculate gamma...");
		long start = System.currentTimeMillis();

		for (int t = 0; t < sequenceLen; t++) {
			for (int i = 0; i < stateNum; i++) {
				gamma[t][i] = alpha[t][i] + beta[t][i];
			}

			double logSum = logSum(gamma[t]);
			for (int j = 0; j < stateNum; j++) {
				gamma[t][j] = gamma[t][j] - logSum;
			}
		}

		long end = System.currentTimeMillis();
		logger.info("calculate finished..., time comsumed: " + (end - start) + "ms");

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
		logger.info("update transfer probability parameters A...");

		double[] gammaSum = new double[stateNum];
		double[] tmp = new double[sequenceLen - 1];
		for (int i = 0; i < stateNum; i++) {
			for (int t = 0; t < sequenceLen - 1; t++) {
				tmp[t] = gamma[t][i];
			}
			gammaSum[i] = logSum(tmp);
		}

		long start1 = System.currentTimeMillis();
		double[] ksiLogProbArr = new double[sequenceLen - 1];
		for (int i = 0; i < stateNum; i++) {
			for (int j = 0; j < stateNum; j++) {
				for (int t = 0; t < sequenceLen - 1; t++) {
					ksiLogProbArr[t] = ksi[t][i][j];
				}
				transferProbability1[i][j] = logSum(ksiLogProbArr) - gammaSum[i];
			}
		}

		long end1 = System.currentTimeMillis();
		logger.info("update finished..., time comsumed: " + (end1 - start1) + "ms");

	}

	protected void updateB(int[] x, double[][] gamma) {
		double[] gammaSum2 = new double[stateNum];
		double[] tmp2 = new double[sequenceLen];
		for (int i = 0; i < stateNum; i++) {
			for (int t = 0; t < sequenceLen; t++) {
				tmp2[t] = gamma[t][i];
			}
			gammaSum2[i] = logSum(tmp2);
		}

		logger.info("update B...");
		long start2 = System.currentTimeMillis();
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
					emissionProbability[i][k] = INFINITY;
					continue;
				}

				double[] validArr = new double[valid.size()];
				for (int q = 0; q < valid.size(); q++) {
					validArr[q] = valid.get(q);
				}

				double validSum = logSum(validArr);

				emissionProbability[i][k] = validSum - gammaSum2[i];
			}
		}

		long end2 = System.currentTimeMillis();
		logger.info("update finished..., time comsumed: " + (end2 - start2) + "ms");

	}

	public double logSum(double[] logProbaArr) {
		if (logProbaArr.length == 0) {
			return INFINITY;
		}

		double max = max(logProbaArr);
		double result = 0;
		for (int i = 0; i < logProbaArr.length; i++) {
			result += Math.exp(logProbaArr[i] - max);
		}
		return max + Math.log(result);
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
			deltas[0][i] = pi[i] + emissionProbability[i][o[0]];
		}
		
		for (int t = 1; t < o.length; t++) {
			for (int i = 0; i < this.stateNum; i++) {
				deltas[t][i] = deltas[t - 1][0] + transferProbability1[0][i];
				for (int j = 1; j < this.stateNum; j++) {
					double tmp = deltas[t - 1][j] + transferProbability1[j][i];
					if (tmp > deltas[t][i]) {
						deltas[t][i] = tmp;
						states[t][i] = j;
					}
				}
				deltas[t][i] += emissionProbability[i][o[t]];
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
}
