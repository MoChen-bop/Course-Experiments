package singleNGramStemmer;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class Test {
	
	private Test() { }
	
	public static void main(String[] args) {
		List<String> stems = SNGStemmer.getNGrams("i", 4);
		for (String stem : stems) {
			System.out.println(stem);
		}
		
		List<String> lexison = new ArrayList<>();

		try {
			FileReader fr = new FileReader("C:\\Users\\dell\\Desktop\\stemmerEval-master\\corpus\\goldLemma\\en\\bnc\\NONAC.bnc.txt.words.txt");
			BufferedReader bf = new BufferedReader(fr);
			String word;
			while((word = bf.readLine()) != null) {
				lexison.add(word.trim());
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
		
		Map<String, String> dictionary = SNGStemmer.SNGStem(lexison, 4);		
		
		for (String word : dictionary.keySet()) {
			System.out.println(word + " --> " + dictionary.get(word));
		}
	}

}
