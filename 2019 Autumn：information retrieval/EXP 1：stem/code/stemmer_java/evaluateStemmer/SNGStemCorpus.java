package evaluateStemmer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;

import singleNGramStemmer.SNGStemmer;


public class SNGStemCorpus {
	
	public static double testSNGStem(String corpusPath, String outputDirectory, String outputFileName) {
		
		System.out.println("Begin extract lexicon...");
		List<String> lexicon = extractLexicon(corpusPath);
		System.out.println("Lexicon's length: " + lexicon.size());
		String outputPath;
		int n;
		
		System.out.println("Begin test n-3 gram stemmer...");
		n = 3;
		outputPath = outputDirectory + "\\n-" + n + "\\";
		double time3 = testSNG(lexicon, corpusPath, outputPath, outputFileName, n);
		
		System.out.println("Begin test n-4 gram stemmer...");
		n = 4;
		outputPath = outputDirectory + "\\n-" + n + "\\";
		double time4 = testSNG(lexicon, corpusPath, outputPath, outputFileName, 4);
		
		System.out.println("Begin test n-5 gram stemmer...");
		n = 5;
		outputPath = outputDirectory + "\\n-" + n + "\\";
		double time5 = testSNG(lexicon, corpusPath, outputPath, outputFileName, 5);
		
		System.out.println("Begin test n-6 gram stemmer...");
		n = 6;
		outputPath = outputDirectory + "\\n-" + n + "\\";
		double time6 = testSNG(lexicon, corpusPath, outputPath, outputFileName, 6);
		

		System.out.println("Begin test n-7 gram stemmer...");
		n = 7;
		outputPath = outputDirectory + "\\n-" + n + "\\";
		double time7 = testSNG(lexicon, corpusPath, outputPath, outputFileName, 7);
		
		return (time3 + time4 + time5 + time6 + time7) / 5;
	}
	
	private static double testSNG(List<String> lexicon, String corpusPath, String outputDirectory, String outputFileName, int n) {
		
		System.out.println("Begin build stemmer...");
		SNGStemmer stemmer = new SNGStemmer(lexicon, n);
		
		File f = new File(outputDirectory);
		if (!f.exists()) {
			f.mkdirs();
		}

		System.out.println("Begin stem...");
		try {
			FileReader fr = new FileReader(corpusPath);
			BufferedReader bf = new BufferedReader(fr);

			File output = new File(outputDirectory, outputFileName);
			FileOutputStream outputStream = new FileOutputStream(output);
			String word;
			int count = 0;
			long startTime = System.currentTimeMillis();
			
			while((word = bf.readLine()) != null) {
				word = word.trim();
				word = word.toLowerCase();
				String stem = stemmer.stem(word);
				String regex = "^-*|-*$";
			    stem = stem.replaceAll(regex, "");
				outputStream.write((word + "\t" + stem + "\n").getBytes());
				count++;
			}
			
			long endTime = System.currentTimeMillis();
			long timeConsumed = endTime - startTime; // ms
			
			bf.close();
			fr.close();
			outputStream.close();
			return (double)timeConsumed / count;			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		return 0;
	}
	
	public static List<String> extractLexicon(String corpusPath) {
		Set<String> lexicon = new HashSet<>();
		try {
			FileReader fr = new FileReader(corpusPath);
			BufferedReader bf = new BufferedReader(fr);
			String word;
			while((word = bf.readLine()) != null) {
				word = word.trim();
				word = word.toLowerCase();
				if (word != "" && !lexicon.contains(word))
					lexicon.add(word);
			}
			
			bf.close();
			fr.close();		
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return new ArrayList<>(lexicon);
	}
}
