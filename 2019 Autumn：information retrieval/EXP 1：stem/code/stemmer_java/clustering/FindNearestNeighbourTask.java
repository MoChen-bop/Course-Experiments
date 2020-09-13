package clustering;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveTask;

public class FindNearestNeighbourTask extends RecursiveTask<List<MinDistancePair>>{

	private static long SEQUENTIAL_THRESHOLD = 100000;
	
	static List<MinDistancePair> findMinDistancePairs(ClusterManager manager, int clusterID) {
		int n = manager.size();
		int cores = Runtime.getRuntime().availableProcessors();
		SEQUENTIAL_THRESHOLD = (long)Math.ceil((double)n / (4.0*cores));
		
		return ClusterManager.commonPool.invoke(new FindNearestNeighbourTask(manager, clusterID, 0, n));
	}
	
	private ClusterManager manager;
	private int clusterID;
	private int start;
	private int end;
	
	private FindNearestNeighbourTask(ClusterManager manager, int clusterID, int start, int end) {
		this.manager = manager;
		this.clusterID = clusterID;
		this.start = start;
		this.end = end;
	}
	
	@Override
	protected List<MinDistancePair> compute() {
		if (end - start <= SEQUENTIAL_THRESHOLD) {
			float minDist = Float.POSITIVE_INFINITY;
			List<MinDistancePair> minDistancePairs = new ArrayList<>();
			MyCustomBigArray distances = manager.dist;
			for (int anotherClusterID = start; anotherClusterID < end; anotherClusterID++) {
				long k;
				if (clusterID < anotherClusterID) {
					k = manager._k(clusterID, anotherClusterID);
					System.out.println("r: " + clusterID + ", s: " + anotherClusterID + ", k: " + k);
				} else if(clusterID > anotherClusterID) {
					k = manager._k(anotherClusterID, clusterID);
					System.out.println("r: " + anotherClusterID + ", s: " + clusterID + ", k: " + k);
				} else
					continue;
				
				if (distances.get(k) < minDist) {
					minDistancePairs = new ArrayList<>();
					minDist = distances.get(k);
					int r, s;
					
					if (clusterID < anotherClusterID) {
						r = clusterID;
						s = anotherClusterID;
					} else {
						r = anotherClusterID;
						s = clusterID;
					}
					minDistancePairs.add(new MinDistancePair(r, s, minDist));
				} else if (distances.get(k) == minDist) {
					int r, s;
					
					if (clusterID < anotherClusterID) {
						r = clusterID;
						s = anotherClusterID;
					} else {
						r = anotherClusterID;
						s = clusterID;
					}
					minDistancePairs.add(new MinDistancePair(r, s, minDist));
				} // end if
			} // end for
			return minDistancePairs;
		} else {
			int mid = start + (end - start) / 2;
			System.out.println("Start: " + start + "End: " + end + "min: " + mid);
			FindNearestNeighbourTask left = new FindNearestNeighbourTask(manager, clusterID, start, mid);
			FindNearestNeighbourTask right = new FindNearestNeighbourTask(manager, clusterID, mid, end);
			
			left.fork();
			
			List<MinDistancePair> rightAns = right.compute();
			List<MinDistancePair> leftAns  = left.join();
			
			if (leftAns.size() == 0) return rightAns;
			if (rightAns.size() == 0) return leftAns;
			
			if (leftAns.get(0).getDist() == rightAns.get(0).getDist()) {
				leftAns.addAll(rightAns);
				return leftAns;
			} else if (leftAns.get(0).compareTo(rightAns.get(0)) == -1) {
				return leftAns;
			} else {
				return rightAns;
			}
		}
	}
	

}
