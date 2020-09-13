package YASS;

import clustering.*;

import java.io.*;
import java.nio.file.DirectoryNotEmptyException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.*;

public class test {

	private test() {
	}

	public static void main(String[] args) {
		List<String> stopwords = new ArrayList<>();
		try {
			File fileDir = new File("C:\\Users\\dell\\Desktop\\stemmerEval-master\\stopwords\\English.snow.txt");
			BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(fileDir), "UTF8"));
			String line;
			while ((line = in.readLine()) != null) {
				stopwords.add(line);
			}
			in.close();
		} catch (Exception e) {
			System.err.println(e.getMessage());
		}

		List<String> lexicon = new ArrayList<>();
		try {
			File fileDir = new File(
					"C:\\Users\\dell\\Desktop\\stemmerEval-master\\corpus\\goldLemma\\en\\bnc\\OTHERPUB.bnc.txt.words.txt");
			BufferedReader in = new BufferedReader(new InputStreamReader(new FileInputStream(fileDir), "UTF8"));
			String line;
			while ((line = in.readLine()) != null) {
				line = line.trim();
				if (stopwords.contains(line)) {
					continue;
				}
				lexicon.add(line);
			}
			in.close();
		} catch (Exception e) {
			System.err.println(e.getMessage());
		}
		
		List<MergeHistoryRecord> mergeHistory = new ArrayList<>();
		DistanceMeasure d = DistanceManager.d1();
		mergeHistory = HierarchicalClustering.calculateClusters(d, lexicon);
		float[] thresholds = {(float) 0.3};
		List<ClusterSet> snapshots = HistoryClusterBuilder.buildSetsFromHistory(lexicon, mergeHistory, thresholds);
		for (ClusterSet cs : snapshots) {
			Map<String, String> stemmedDict = YASS.stemFromClusterSet(cs);
			System.out.println();
			System.out.println();
			System.out.println();
			System.out.println(cs.getThreshold());
			Object[] keys = stemmedDict.keySet().toArray();
			Arrays.sort(keys);
			for (Object key : keys) {
				String k = (String) key;
				System.out.println(k + "\t" + stemmedDict.get(k));
			}
		}
	}

}
