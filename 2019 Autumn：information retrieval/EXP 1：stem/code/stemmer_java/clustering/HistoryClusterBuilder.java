package clustering;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class HistoryClusterBuilder {

	public static List<ClusterSet> buildSetsFromHistory(List<String> words, List<MergeHistoryRecord> history, float[] thresholds) {
		List<ClusterSet> snapshots = new ArrayList<>();
		ClusterSet clusterSet = new ClusterSet(0);
		
		for (int i = 0; i < words.size(); i++) {
			List<String> cw = new ArrayList<>();
			cw.add(words.get(i));
			clusterSet.addCluster(new Cluster(i, cw));
		}
		int nextId = words.size();
		
		Arrays.parallelSort(thresholds);
		int cntThreshold = 0;
		
		for (MergeHistoryRecord record : history) {
			if (cntThreshold < thresholds.length && record.getDist() > thresholds[cntThreshold]) {
				ClusterSet newSet = clusterSet.copy();
				newSet.setThreshold(thresholds[cntThreshold]);
				snapshots.add(newSet);
				cntThreshold++;
			}
			
			Cluster cluster1 = clusterSet.getCluster(record.getC1());
			Cluster cluster2 = clusterSet.getCluster(record.getC2());
			
			Cluster merged = Cluster.merge(nextId,  cluster1,  cluster2);
			nextId++;
			clusterSet.removeCluster(record.getC1());
			clusterSet.removeCluster(record.getC2());
			clusterSet.addCluster(merged);
		}
		
		return snapshots;
	}
	
	public static ClusterSet buildSetsFromHistory(List<String> words, List<MergeHistoryRecord> history, float threshold) {
		ClusterSet clusterSet = new ClusterSet(0);
		
		for (int i = 0; i < words.size(); i++) {
			List<String> cw = new ArrayList<>();
			cw.add(words.get(i));
			clusterSet.addCluster(new Cluster(i, cw));
		}
		int nextId = words.size();
		
		for (MergeHistoryRecord record : history) {
			if (record.getDist() > threshold) {
				break;
			}
			
			Cluster cluster1 = clusterSet.getCluster(record.getC1());
			Cluster cluster2 = clusterSet.getCluster(record.getC2());
			
			Cluster merged = Cluster.merge(nextId,  cluster1,  cluster2);
			nextId++;
			clusterSet.removeCluster(record.getC1());
			clusterSet.removeCluster(record.getC2());
			clusterSet.addCluster(merged);
		}
		
		return clusterSet;
	}
}
