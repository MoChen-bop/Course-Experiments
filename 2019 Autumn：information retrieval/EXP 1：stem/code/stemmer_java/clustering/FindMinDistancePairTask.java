package clustering;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.RecursiveTask;

public class FindMinDistancePairTask extends RecursiveTask<List<MinDistancePair>> {
	
	private static long SEQUENTIAL_THRESHOLD = 100000;
	
	public static List<MinDistancePair> findMinDistancePairs(ClusterManager manager) {
		long n = manager.size();
		long last = (n * (n-1)) / (long)2;
		int cores = Runtime.getRuntime().availableProcessors();
		SEQUENTIAL_THRESHOLD = (long)Math.ceil((double) last / (4.0 * cores));
		return ClusterManager.commonPool.invoke(new FindMinDistancePairTask(manager, 0, last));
	}
	
	private ClusterManager manager;
	private long start;
	private long end;
	
	private FindMinDistancePairTask(ClusterManager manager, long start, long end) {
		this.manager = manager;
		this.start = start;
		this.end = end;
	}

	@Override
	protected List<MinDistancePair> compute() {
		if (end - start <= SEQUENTIAL_THRESHOLD) {
			float minDist = Float.POSITIVE_INFINITY;
			List<MinDistancePair> minDistancePairs = new ArrayList<>();
			MyCustomBigArray distances = manager.dist;
			for (long k = start; k < end; k++) {
				if (distances.get(k) < minDist) {
					minDistancePairs = new ArrayList<>();
					minDist = distances.get(k);
					int r = manager._i(k);
					int s = manager._j(k);
					
					if (s < r) {
						int t = r;
						r = s; 
						s = t;
					}
					
					minDistancePairs.add(new MinDistancePair(r, s, minDist));
				} else if (distances.get(k) == minDist) {
					minDist = distances.get(k);
					int r = manager._i(k);
					int s = manager._j(k);
					
					if (s < r) {
						int t = r;
						r = s;
						s = t;
					}
					
					minDistancePairs.add(new MinDistancePair(r, s, minDist));
				}
			}
			return minDistancePairs;
		} else {
			long mid = start + (end - start) / 2;
			FindMinDistancePairTask left = new FindMinDistancePairTask(manager, start, mid);
			FindMinDistancePairTask right = new FindMinDistancePairTask(manager, mid, end);
			left.fork();
			
			List<MinDistancePair> rightAns = right.compute();
			List<MinDistancePair> leftAns = left.join();
			
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
