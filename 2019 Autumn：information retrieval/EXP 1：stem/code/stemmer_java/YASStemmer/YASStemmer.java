package YASStemmer;

import YASS.DistanceMeasure;
import YASS.YASS;
import clustering.ClusterSet;
import clustering.HierarchicalClustering;
import clustering.HistoryClusterBuilder;
import clustering.MergeHistoryRecord;

import java.nio.file.DirectoryStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class YASStemmer {

	private Map<String, String> dictory;
	private Map<Float, Map<String, String>> dictorys;
	
	public YASStemmer(List<String> lexicon, DistanceMeasure d, float threshold) {
		List<MergeHistoryRecord> mergeHistory = new ArrayList<>();
		mergeHistory = HierarchicalClustering.calculateClusters(d, lexicon);
		ClusterSet cs = HistoryClusterBuilder.buildSetsFromHistory(lexicon, mergeHistory, threshold);
		this.dictory = YASS.stemFromClusterSet(cs);
	}
	
	public YASStemmer(List<String> lexicon, DistanceMeasure d, float[] thresholds) {
		List<MergeHistoryRecord> mergeHistory = new ArrayList<>();
		System.out.println("Clustering...");
		mergeHistory = HierarchicalClustering.calculateClusters(d, lexicon);
		System.out.println("Building cluster...");
		List<ClusterSet> cs = HistoryClusterBuilder.buildSetsFromHistory(lexicon, mergeHistory, thresholds);
		this.dictorys = new HashMap<>();
		for (int i = 0; i < cs.size(); i++) {
			Map<String, String> dic = YASS.stemFromClusterSet(cs.get(i));
			this.dictorys.put(cs.get(i).getThreshold(), dic);
		}
	}
	
	public String stem(String word) {
		if (dictory.containsKey(word))
			return dictory.get(word);
		else return word;
	}
	
	public String stem(String word, float threshold) {
		if (this.dictorys.get(threshold).containsKey(word)) {
			return this.dictorys.get(threshold).get(word);
		}
		else return word;
	}
}
