package HMMStemmer;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

public class HMMStemmer {
	private HMM model;
	private CorpusIterator iterator;
	private int printInfoInterval = 2;
	private int[] buffer = new int[50];
	public int splitPoint;
	public static final double ZERO = 0;
	public static final int BEGINTAG = 26;
	public static final int ENDTAG = 27;
	
	public HMMStemmer(CorpusIterator iter, int prefixNum, int suffixNum) {
		int stateNum = prefixNum + suffixNum;
		this.splitPoint = prefixNum;
		int observationNum = 26 + 2;
		double[] pi = new double[stateNum];
		double[][] transferProbability = new double[stateNum][stateNum];
		double[][] emissionProbability = new double[stateNum][observationNum];
		
		for (int i = 0; i < splitPoint; i++) {
			pi[i] = 1;
		}
		for (int i = splitPoint; i < stateNum; i++) {
			pi[i] = ZERO;
		}
		
		for (int i = 0; i < stateNum; i++) {
			for (int j = 0; j < stateNum; j++) {
				transferProbability[i][j] = ZERO;
			}
		}
		
		for (int i = 0; i < stateNum - 1; i++) {
			transferProbability[i][i+1] = 1;
		}
		for (int i = 1; i < splitPoint; i++) {
			transferProbability[i][i] = 1;
		}
		transferProbability[stateNum - 1][stateNum - 1] = 1;
		/*transferProbability = new double[][]{
			{ZERO,    1, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO},
			{ZERO,    1,    1, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO},
			{ZERO, ZERO,    1,    1, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO},
			{ZERO, ZERO, ZERO,    1,    1, ZERO, ZERO, ZERO, ZERO, ZERO},
			{ZERO, ZERO, ZERO, ZERO,    1,    1, ZERO, ZERO, ZERO, ZERO},
			{ZERO, ZERO, ZERO, ZERO, ZERO, ZERO,    1, ZERO, ZERO, ZERO},
			{ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO,    1, ZERO, ZERO},
			{ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO,    1, ZERO},
			{ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO,    1},
			{ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO,    1},
		};*/
		for (int i = 0; i < emissionProbability.length; i++) 
			for (int j = 0; j < emissionProbability[0].length; j++)
				emissionProbability[i][j] = 1;
		for (int i = splitPoint; i < emissionProbability.length; i++)
			emissionProbability[i][26] = ZERO;
		for (int i = 0; i < splitPoint; i++)
			emissionProbability[i][27] = ZERO;
		model = new HMM(stateNum, observationNum, pi, transferProbability, emissionProbability);
		iterator = iter;
	}
	
	public void train(int maxIter, double precision) {
		double loss = 0;
		for (int i = 1; i <= maxIter; i++) {
			iterator.reset();
			while (iterator.hasNext()) {
				Word word = iterator.getNextWord();
				int[] code = word.getCode();
				model.baumWelch(code);
			}
			
			loss = model.modifyLambda();
			if (i % printInfoInterval == 0) {
				System.out.println("[" + i + "/" + maxIter + "] loss = " + loss);
			}
			//model.printInfo();

			loss = 0;
		}
	}
	
	public int stem(char s[], int len) {
		Word word = new Word(len + 2);
		word.add(s, len);
		int[] path = model.mostLikely(word.getCode());
		int endPoint = 0;
		while (endPoint < len && path[endPoint] < splitPoint)
			endPoint++;
		return endPoint;
	}
	
	public String stem(String word) {
		String regex = "^[a-z]+$";
		if (!word.matches(regex)) return word;
		
		int len = word.length();
		char[] s = word.toCharArray();
		int endPoint = stem(s, len);
		return String.valueOf(s, 0, endPoint);
	}
	
	public void saveModel(String outputPath, String outputFileName) {
		File f = new File(outputPath);
		if (!f.exists()) {
			f.mkdirs();
		}
		File output = new File(outputPath, outputFileName);
		try {
			FileOutputStream outputStream = new FileOutputStream(output);
			outputStream.write("Pi: \n".getBytes());
			for (int i = 0; i < model.pi.length; i++) {
				outputStream.write((String.valueOf(model.pi[i]) + "\t").getBytes());
			}
			outputStream.write("\n".getBytes());
			outputStream.write("TransferProbability: \n".getBytes());
			for (int i = 0; i < model.transferProbability.length; i++) {
				for (int j = 0; j < model.transferProbability[0].length; j++) {
					outputStream.write((String.valueOf(model.transferProbability[i][j]) + "\t").getBytes());
				}
				outputStream.write("\n".getBytes());
			}
			outputStream.write("EmissionProbability: \n".getBytes());
			for (int i = 0; i < model.emissionProbability.length; i++) {
				for (int j = 0; j < model.emissionProbability[0].length; j++) {
					outputStream.write((String.valueOf(model.emissionProbability[i][j]) + "\t").getBytes());
				}
				outputStream.write("\n".getBytes());
			}
			outputStream.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
	}
}
