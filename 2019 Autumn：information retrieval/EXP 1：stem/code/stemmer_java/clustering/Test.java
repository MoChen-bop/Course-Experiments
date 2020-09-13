package clustering;

import java.util.List;
import java.util.ArrayList;

public class Test {
	
	private Test() { }
	
	public static void main(String[] args) {
		List<Cluster> clusters = new ArrayList<>();
		for (int i = 0; i < 8; i++) {
			List<String> words = new ArrayList<>();
			words.add("w" + String.valueOf(i + 1));
			clusters.add(new Cluster(i, words));
		}
		ClusterManager manager = new ClusterManager(clusters);
		for (int i = 0; i < manager.size(); i++) {
			for (int j = i + 1; j < manager.size(); j++) {
				long k = manager._k(i, j);
				manager.dist.set(k, 100);
			}
		}
		
		long k;
		k = manager._k(0, 1);
		manager.dist.set(k, 1);
		k = manager._k(1, 2);
		manager.dist.set(k, 2);
		k = manager._k(1, 3);
		manager.dist.set(k, 3);
		k = manager._k(1, 4);
		manager.dist.set(k, 2);
		k = manager._k(2, 3);
		manager.dist.set(k, 4);
		k = manager._k(3, 4);
		manager.dist.set(k, 2);
		k = manager._k(3, 6);
		manager.dist.set(k, 11);
		k = manager._k(5, 6);
		manager.dist.set(k, 2);
		k = manager._k(5, 7);
		manager.dist.set(k, 4);
		k = manager._k(6, 7);
		manager.dist.set(k, 3);
		
		ClusterSet clusterSet = NearestNeighbourClustering.clustering(manager);
		clusterSet.print();
	}

}
