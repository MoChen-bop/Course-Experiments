package evaluateStemmer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import YASS.DistanceManager;
import YASS.DistanceMeasure;
import YASStemmer.YASStemmer;

public class YASStemCorpus {
	
	public static final int MAXNUM = 20000;
	
	public static double testYASStem(String corpusPath, String outputDirectory, String fileName) {
		System.out.println("Building lexicon...");
		List<String> lexicon = extractLexicon(corpusPath);
		System.out.println("Lexicon's size: " + String.valueOf(lexicon.size()));
		
		System.out.println("Begin test D1...");
		DistanceMeasure d1 = DistanceManager.d1();
		float[] thresholds1 = {(float) 0.05, (float) 0.1, (float) 0.15, (float) 0.2, (float) 0.25, (float)0.3};
		YASStemmer stemmer = new YASStemmer(lexicon, d1, thresholds1);
		double time1 = beginTest(stemmer, corpusPath, outputDirectory, fileName, lexicon, d1, thresholds1);
		
		System.out.println("Begin test D2...");
		DistanceMeasure d2 = DistanceManager.d2();
		float[] thresholds2 = {(float) 0.1, (float) 0.2, (float) 0.3, (float) 0.4, (float) 0.5, (float)0.6};
		stemmer = new YASStemmer(lexicon, d2, thresholds2);
		double time2 = beginTest(stemmer, corpusPath, outputDirectory, fileName, lexicon, d2, thresholds2);
		
		System.out.println("Begin test D3...");
		DistanceMeasure d3 = DistanceManager.d3();
		float[] thresholds3 = {(float) 0.5, (float) 1, (float) 1.5, (float) 2, (float) 2.5, (float)3};
		stemmer = new YASStemmer(lexicon, d3, thresholds3);
		double time3 = beginTest(stemmer, corpusPath, outputDirectory, fileName, lexicon, d3, thresholds3);
		
		System.out.println("Begin test D4...");
		DistanceMeasure d4 = DistanceManager.d4();
		float[] thresholds4 = {(float) 0.2, (float) 0.4, (float) 0.6, (float) 0.8, (float) 1, (float)1.2};
		stemmer = new YASStemmer(lexicon, d4, thresholds4);
		double time4 = beginTest(stemmer, corpusPath, outputDirectory, fileName, lexicon, d4, thresholds4);
		
		/*System.out.println("Begin test D5-3...");
		DistanceMeasure d53 = DistanceManager.d5(3);
		float[] thresholds5 = {(float) 0.5, (float) 0.6, (float) 0.7, (float) 0.8, (float)0.9, (float)1};
		stemmer = new YASStemmer(lexicon, d53, thresholds5);
		double time53 = beginTest(stemmer, corpusPath, outputDirectory, fileName, lexicon, d53, thresholds5);
		
		System.out.println("Begin test D5-4...");
		DistanceMeasure d54 = DistanceManager.d5(4);
		stemmer = new YASStemmer(lexicon, d54, thresholds5);
		double time54 = beginTest(stemmer, corpusPath, outputDirectory, fileName, lexicon, d54, thresholds5);
		
		System.out.println("Begin test D5-5...");
		DistanceMeasure d55 = DistanceManager.d5(55);
		stemmer = new YASStemmer(lexicon, d55, thresholds5);
		double time55 = beginTest(stemmer, corpusPath, outputDirectory, fileName, lexicon, d55, thresholds5);
		*/
		return (time1 + time2 + time3 + time4) / 4;
				//+ time53 + time54 + time55) / 7;
	}
	
	public static boolean isWord(String word) {
		for (int i = 0; i < word.length(); i++) {
			if (!Character.isLetter(word.charAt(i))) return false;
		}
		return true;
		
	}
	
	public static List<String> extractStopWords(String stopwordsPath) {
		List<String> stopwords = new ArrayList<String>();
		FileReader fr;
		try {
			fr = new FileReader(stopwordsPath);
			BufferedReader bf = new BufferedReader(fr);
			String word;
			while((word = bf.readLine()) != null) {
				word = word.trim();
				word = word.toLowerCase();
				stopwords.add(word);
			}
			
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			e.printStackTrace();
		}
		return stopwords;
	}

	public static List<String> extractLexicon(String corpusPath) {
		Map<String, Integer> lexicon = new HashMap<>();
		List<String> topLexiconList = new ArrayList<>();
		
		String stopwordsPath = "C:\\Users\\dell\\Desktop\\stemmerEval-master\\stopwords\\English.snow.txt";
		List<String> stopwords = new ArrayList<String>();
		stopwords = extractStopWords(stopwordsPath);
		int count = 0;
		try {
			FileReader fr = new FileReader(corpusPath);
			BufferedReader bf = new BufferedReader(fr);
			String word;
			FileOutputStream output = new FileOutputStream(corpusPath + ".lexicon.txt");
			while((word = bf.readLine()) != null) {
				word = word.trim();
				word = word.toLowerCase();
				if (word.length() > 2 && isWord(word) && !stopwords.contains(word)) {
					if (lexicon.containsKey(word))
						lexicon.replace(word, lexicon.get(word) +1);
					else lexicon.put(word, 1);
					count++;
				}
			}
			
			List<Map.Entry<String, Integer>> lexiconEntry = new ArrayList<Map.Entry<String, Integer>>(lexicon.entrySet());
			lexiconEntry.sort(new Comparator<Map.Entry<String, Integer>>() {

				@Override
				public int compare(Entry<String, Integer> o1, Entry<String, Integer> o2) {
					return o2.getValue().compareTo(o1.getValue());
				}
				
			});
			int index = 0;
			output.write(("total size: " + lexiconEntry.size() + "\n").getBytes());
			while(index < lexiconEntry.size() && index < MAXNUM) {
				topLexiconList.add(lexiconEntry.get(index).getKey());
				output.write((lexiconEntry.get(index).getKey() + "\t" 
						+ ((double)(lexiconEntry.get(index).getValue()) / count)
						+ "\n").getBytes());
				index++;
			}
			bf.close();
			fr.close();		
			output.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return topLexiconList;
	}
	
	public static double beginTest(YASStemmer stemmer, String corpusPath, String outputPath, String outputFileName,
			List<String> lexicon, DistanceMeasure d, float[] thresholds) {
		double averageConsumedTime = 0;
		String outputDirectory;
		for (float threshold : thresholds) {
			try {
				FileReader fr = new FileReader(corpusPath);
				BufferedReader bf = new BufferedReader(fr);
				
				outputDirectory = outputPath + "\\" + d.getName() + "\\threshold-" + String.valueOf(threshold);
				File f = new File(outputDirectory);
				if (!f.exists()) {
					f.mkdirs();
				}
				File output = new File(outputDirectory, outputFileName);
				FileOutputStream outputStream = new FileOutputStream(output);
				String word;
				int count = 0;
				long startTime = System.currentTimeMillis();
				
				while((word = bf.readLine()) != null) {
					word = word.trim();
					word = word.toLowerCase();
					String stem = stemmer.stem(word, threshold);
					outputStream.write((word + "\t" + stem + "\n").getBytes());
					count++;
				}
				
				long endTime = System.currentTimeMillis();
				long timeConsumed = endTime - startTime; // ms
				
				bf.close();
				fr.close();
				outputStream.close();
				
				averageConsumedTime += (double)timeConsumed / count / thresholds.length;
				
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		return averageConsumedTime;
	}
}
