package clustering;

import YASS.DistanceMeasure;

import java.util.*;

public class HierarchicalClustering {

	public static List<MergeHistoryRecord> calculateClusters(DistanceMeasure d, List<String> words) {
		int n = words.size();
		int printInterval = (int)Math.max(10,  n * 0.00005);
		
		List<Cluster> clusters = new ArrayList<>();
		for (int i = 0; i < words.size(); i++) {
			List<String> clusterWords = new ArrayList<>();
			clusterWords.add(words.get(i));
			clusters.add(new Cluster(i, clusterWords));
		}

		return clusterer(d, clusters, n);
	}

	private static List<MergeHistoryRecord> clusterer(DistanceMeasure d, List<Cluster> clusters, int nextId) {
		int printInterval = (int)Math.max(10,  clusters.size() * 0.00005);
		ClusterManager manager = new ClusterManager(clusters, d);
		int cntIter = 0;
		List<MergeHistoryRecord> historyRecords = new ArrayList<>();
		while (manager.size() != 1) {
			List<MinDistancePair> minDistancePairs = manager.findMinDistancePairs();
			Set<Integer> mergedCluster = new HashSet<>();
			List<Cluster> newClusters = new ArrayList<>();
			
			for (MinDistancePair pair : minDistancePairs) {
				int r = pair.getR();
                int s = pair.getS();
                if (mergedCluster.contains(r) || mergedCluster.contains(s)){
                    continue;
                }
                newClusters.add(Cluster.merge(nextId, manager.getCluster(r), manager.getCluster(s)));
                historyRecords.add(new MergeHistoryRecord(
                        manager.getCluster(r).getId(),
                        manager.getCluster(s).getId(),
                        nextId,
                        pair.getDist(),
                        manager.size() - newClusters.size())
                );
                mergedCluster.add(r);
                mergedCluster.add(s);
                nextId++;
			}

			List<Integer> toDelete = new ArrayList<>();
            toDelete.addAll(mergedCluster);
            manager.deleteClusters(toDelete);

            for (Cluster c : newClusters) {
                manager.insert(c);
            }

            manager.resize();
            cntIter++;
		}
		return historyRecords;
	}
}
