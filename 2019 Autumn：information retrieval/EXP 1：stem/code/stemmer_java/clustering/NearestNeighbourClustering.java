package clustering;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class NearestNeighbourClustering {
	public static ClusterSet clustering(ClusterManager manager) {

		Map<Integer, Integer> wordID2ClusterID = new HashMap<>();
		int cID = 0;
		ClusterSet clusterSet = new ClusterSet(0);
		for (int i = 0; i < manager.size(); i++) {
			String word = manager.getCluster(i).getWords().get(0);
			List<String> words = new ArrayList<>();
			words.add(word);
			clusterSet.addCluster(new Cluster(cID, words));
			wordID2ClusterID.put(i, cID);
			System.out.println("put: " + i + ", " + cID);
			cID++;
		}
		
		for (int clusterID = 0; clusterID < manager.size(); clusterID++) {
			System.out.println("ClucterID: " + clusterID);
			List<MinDistancePair> minDistancePairs = manager.findNearestNeighbours(clusterID);
			System.out.println("END");
			for (MinDistancePair pair : minDistancePairs) {
				System.out.println("Find min distance Pair: " + pair.getDist());
				int wordIDR = pair.getR();
				int wordIDS = pair.getS();
				
				int clusterIDR = wordID2ClusterID.get(wordIDR);
				int clusterIDS = wordID2ClusterID.get(wordIDS);
				
				System.out.println("word ID R: " + wordIDR + " , word ID S: " + wordIDS);
				System.out.println("cluster ID R: " + clusterIDR + " , cluster ID S: " + clusterIDS);
				Cluster clusterR = clusterSet.getCluster(clusterIDR);
				Cluster clusterS = clusterSet.getCluster(clusterIDS);
				clusterSet.addCluster(Cluster.merge2(cID, clusterR, clusterS));
				System.out.println("Add cluster:" + cID);
				clusterSet.removeCluster(clusterIDR);
				clusterSet.removeCluster(clusterIDS);
				System.out.println("Remove cluster: " + clusterIDR + " " + clusterIDS);
				
				System.out.println("Replace: " + wordIDR + ", " + cID);
				System.out.println("Replace: " + wordIDS + ", " + cID);
				for (Integer key : wordID2ClusterID.keySet()) {
					if (wordID2ClusterID.get(key) == clusterIDR || 
							wordID2ClusterID.get(key) == clusterIDS)
						wordID2ClusterID.replace(key, cID);
		        }				
				cID++;
			}
		}

		return clusterSet;
	}
}
