package HMMStemmer;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UnsupportedEncodingException;

public class CorpusIterator {
	private File[] file_list;
	private InputStreamReader in;
	private BufferedReader reader;
	private char[] buffer;
	private int endIndex;
	private int currentFileIndex;

	public CorpusIterator(String path) {
		try {
			File current_file = new File(path);
			if (current_file.isDirectory()) {
				file_list = current_file.listFiles();
			} else {
				file_list = new File[] { current_file };
			}
			currentFileIndex = 0;
			in = new InputStreamReader(new FileInputStream(
					file_list[currentFileIndex]), "GBK");
			reader = new BufferedReader(in);
		} catch (UnsupportedEncodingException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		buffer = new char[50];
	}

	public boolean hasNext() {
		int ch = 0;
		endIndex = 0;
		try {
			while ((ch = reader.read()) != -1) {
				if (((int)'a' <= ch && ch <= (int)'z')
						|| ((int)'A' <= ch && ch <= (int)'Z'))
					buffer[endIndex++] = (char) Character.toLowerCase(ch);
				
				if (!Character.isLetter(ch) && endIndex <= 3) {
					endIndex = 0;
				}
				
				if (endIndex == 50 || (!Character.isLetter(ch) && endIndex != 0))
					return true;
			}
			
			if (this.currentFileIndex < this.file_list.length - 1) {
				in.close();
				reader.close();
				this.currentFileIndex++;
				in = new InputStreamReader(new FileInputStream(
						file_list[currentFileIndex]), "GBK");
				reader = new BufferedReader(in);
				
				while ((ch = reader.read()) != -1) {
					if (Character.isLetter(ch))
						buffer[endIndex++] = (char) Character.toLowerCase(ch);
					if (!Character.isLetter(ch) && endIndex != 0)
						return true;
				}
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();

		}

		return false;
	}

	public int getWordLength() {
		return endIndex;
	}

	public Word getNextWord() {
		Word word = new Word(endIndex + 2);
		word.add(buffer, endIndex);
		return word;
	}

	public void reset() {
		try {
			in.close();
			reader.close();
			this.currentFileIndex = 0;
			
			in = new InputStreamReader(new FileInputStream(
					file_list[currentFileIndex]), "GBK");
			reader = new BufferedReader(in);
			
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
