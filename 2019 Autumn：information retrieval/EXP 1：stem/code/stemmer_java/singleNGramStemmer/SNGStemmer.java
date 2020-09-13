package singleNGramStemmer;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SNGStemmer {
	
	private Map<String, String> word2Stem;
	
	public SNGStemmer(List<String> lexicon, int n) {
		word2Stem = SNGStem(lexicon, n);
	}
	
	public String stem(String word) {
		if (word2Stem.containsKey(word)) 
			return word2Stem.get(word);
		else return word;
	}

	public static Map<String, String> SNGStem(List<String> lexison, int n) {
		Map<String, List<String>> stemPostingList = buildPostingList(lexison, n);
		Map<String, Double> stemIDF = calculateStemIDF(stemPostingList, lexison.size());
		Map<String, String> word2Stem = new HashMap<>();
		for (String word : lexison) {
			List<String> stems = getNGrams(word, n);
			String stem = stems.get(0);
			Double maxIDF = stemIDF.get(stem);
			for (String s : stems) {
				if (maxIDF < stemIDF.get(s)) {
					maxIDF = stemIDF.get(s);
					stem = s;
				}
			}
			word2Stem.put(word, stem);
		}
		
		return word2Stem;
	}
	
	private static Map<String, List<String>> buildPostingList(List<String> lexison, int n) {
		Map<String, List<String>> stemPostingList = new HashMap<>();
		
		for (String word : lexison) {
			List<String> stems = getNGrams(word, n);
			for (String stem : stems) {
				if (stemPostingList.containsKey(stem)) {
					if (!stemPostingList.get(stem).contains(word)) {
						stemPostingList.get(stem).add(word);
					}
				} else {
					List<String> newWordList = new ArrayList<>();
					newWordList.add(word);
					stemPostingList.put(stem, newWordList);
				}
			}
		}
		return stemPostingList;
	}
	
	private static Map<String, Double> calculateStemIDF(Map<String, List<String>> stemPostingList, int lexisonLength) {
		Map<String, Double> stemIDF = new HashMap<>();
		for (String stem : stemPostingList.keySet()) {
			double IDF = - Math.log((double)stemPostingList.get(stem).size() / (double)lexisonLength);
			
			stemIDF.put(stem, IDF);
		}
		return stemIDF;
	}
	
	public static List<String> getNGrams(String word, int n) {
		List<String> grams = new ArrayList<>();
//		if (word.length() <= n) {
//			grams.add(word);
//			return grams;
//		}
		
		for (int begin = -n + 1; begin < word.length(); begin++) {
			String gram = getGram(word, begin, n);
			grams.add(gram);
		}
		
		return grams;
	}

	public static String getGram(String word, int begin, int n) {
		int left = begin;
		int right = begin + n;
		String stem = "";
		while (left < 0) {
			stem += "-";
			left++;
		}
		
		if (right < word.length()) {
			stem += word.substring(left, right);
		} else {
			stem += word.substring(left);
			left += word.length() - left;
			while (left < right) {
				stem += "-";
				left++;
			}
		}
		return stem;
	}
}
