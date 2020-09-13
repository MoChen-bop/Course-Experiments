package evaluateStemmer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;

import PorterStemmer.PorterStemmer;

public class PorterStemCorpus {
	
	public static double testPorterStem(String corpusPath, String outputDirectory, String outputFileName) {
		PorterStemmer stemmer = new PorterStemmer();
		try {
			FileReader fr = new FileReader(corpusPath);
			BufferedReader bf = new BufferedReader(fr);
			
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
				for (int i = 0; i < word.length(); i++) {
					stemmer.add(word.charAt(i));
				}
				stemmer.stem();
				String stem = stemmer.toString();
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

}
