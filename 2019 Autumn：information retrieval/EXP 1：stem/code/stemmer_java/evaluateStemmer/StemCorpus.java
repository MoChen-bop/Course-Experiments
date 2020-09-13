package evaluateStemmer;

import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;

public class StemCorpus {
	public static void main(String[] args) {
		String[] corpusPaths = {
				"G:\\dataset\\corpus\\BNC\\Corpus\\ACPROSE.bnc.words.txt",
				"G:\\dataset\\corpus\\BNC\\Corpus\\CONVRSN.bnc.words.txt",
				"G:\\dataset\\corpus\\BNC\\Corpus\\FICTION.bnc.words.txt",
				"G:\\dataset\\corpus\\BNC\\Corpus\\NEWS.bnc.words.txt",
				"G:\\dataset\\corpus\\BNC\\Corpus\\NONAC.bnc.words.txt",
				"G:\\dataset\\corpus\\BNC\\Corpus\\OTHERPUB.bnc.words.txt",
				"G:\\dataset\\corpus\\BNC\\Corpus\\OTHERSP.bnc.words.txt",
				"G:\\dataset\\corpus\\BNC\\Corpus\\UNPUB.bnc.words.txt",
				};
		String outputDirectory = "G:\\dataset\\corpus\\BNC\\Stem";
		//String stemmerName = "Porter";
		//String stemmerName = "Lovins";
		//String stemmerName = "YASS";
		String stemmerName = "SNG";
		String outputPath = outputDirectory + "\\" + stemmerName + "\\" ;
		double averageTime = 0;
		for (String corpusPath: corpusPaths) {
			String domain = corpusPath.split("\\\\")[5].split("\\.")[0];
			String outputFileName = domain + ".bnc.stem.txt";
			System.out.println(outputPath + outputFileName);
			//double consumedTime = PorterStemCorpus.testPorterStem(corpusPath, outputPath, outputFileName);
			//double consumedTime = LovinsStemCorpus.testLovinsStem(corpusPath, outputPath, outputFileName);
			//double consumedTime = HMMStemCorpus.testHMMStem(corpusPath, outputPath, outputFileName);
			//double consumedTime = YASStemCorpus.testYASStem(corpusPath, outputPath, outputFileName);
			double consumedTime = SNGStemCorpus.testSNGStem(corpusPath, outputPath, outputFileName);
			averageTime += consumedTime / corpusPaths.length;
		}
		
		FileOutputStream timeRecorderFile;
		try {
			timeRecorderFile = new FileOutputStream(outputPath + "timeConsumed.txt");
			timeRecorderFile.write(("Average Time per Stem: " + averageTime + "ms").getBytes());
			timeRecorderFile.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
