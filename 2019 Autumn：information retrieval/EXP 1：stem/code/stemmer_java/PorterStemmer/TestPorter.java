package PorterStemmer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;

public class TestPorter {
	
	private TestPorter() { }
	
	public static void main(String[] args)  {
		String path = "C:\\Users\\dell\\Desktop\\stemmerEval-master\\corpus\\goldLemma\\en\\Corpus\\NONAC.bnc.words.txt";
		PorterStemmer stemmer = new PorterStemmer();
		try {
			FileReader fr = new FileReader(path);
			BufferedReader bf = new BufferedReader(fr);
			File file = new File("C:\\Users\\dell\\Desktop\\stemmerEval-master\\corpus\\goldLemma\\en\\Corpus\\words_stem.txt");
			FileOutputStream outputStream = new FileOutputStream(file);
			String word;
			while((word = bf.readLine()) != null) {
				word = word.trim();
				word = word.toLowerCase();
				for (int i = 0; i < word.length(); i++) {
					stemmer.add(word.charAt(i));
				}
				stemmer.stem();
				String stem = stemmer.toString();
				outputStream.write((word + "\t" + stem + "\n").getBytes());
			}
			bf.close();
			fr.close();
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
