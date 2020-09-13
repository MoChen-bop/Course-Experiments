package clustering;

import YASS.DistanceMeasure;

import java.util.*;
import java.util.concurrent.ForkJoinPool;

public class ClusterManager {

	public static ForkJoinPool commonPool = new ForkJoinPool();
	
	private List<Cluster> clusters;
	private DistanceMeasure d;
	MyCustomBigArray dist;
	
	public long _k(int i, int j) {
		long n = clusters.size();
		long k = (n * (n - 1)) / 2 - ( (n-i) * (n - i - 1)) / 2 + j - i - 1;
		k = ( n*(n-1) ) / 2 - 1 - k;
		return k;
	}
	
	public int _i(long k) {
		long n = clusters.size();
		k = ( n*(n-1) ) / 2 - 1 - k;
		long i = n - 2 - (int)Math.floor(Math.sqrt(-8*k + 4*n*(n - 1) - 7)/2 - 0.5);
		return (int)i;
	}
	
	public int _j(long k) {
		long n = clusters.size();
		int i = _i(k);
		k = (n*(n-1)) / 2 - 1 - k;
		long j = (k + i + 1 - (n*(n-1)) / 2 + ((n - i) * ((n-i) - 1)) / 2);
		return (int)j;
	}
	
	public ClusterManager(List<Cluster> clusters) {
		this.clusters = clusters;
		this.d = d;
		long n = clusters.size();
		long tot = (n*(n-1)) / 2;
		dist = new MyCustomBigArray(tot);
	}
	
	public ClusterManager(List<Cluster> clusters, DistanceMeasure d) {
		this.clusters = clusters;
		this.d = d;
		long n = clusters.size();
		long tot = (n*(n-1)) / 2;
		dist = new MyCustomBigArray(tot);
		
		BuildDistanceMatrixTask.buildDistanceMatrix(this, d);
	}
	
	public void deleteClusters(List<Integer> indexes) {
		Collections.sort(indexes);
		
		int start;
		int CHUNK_SIZE = 7;
		for (start = 0; start + CHUNK_SIZE < indexes.size(); start += CHUNK_SIZE) {
			List<Integer> ar = new ArrayList<>();
			for (Integer i : indexes.subList(start,  start + CHUNK_SIZE)) {
				ar.add(i);
			}
			actuallyDeleteClusters(ar);
			for (int q = start + CHUNK_SIZE; q < indexes.size(); q++) {
				indexes.set(q,  indexes.get(q) - CHUNK_SIZE);
			}
		}
		List<Integer> ar = new ArrayList<>();
		for (Integer i : indexes.subList(start,  indexes.size())) {
			ar.add(i);
		}
		
		if(ar.size() > 0)
			actuallyDeleteClusters(ar);
	}
	
	private void actuallyDeleteClusters(List<Integer> indexes) {
		int n = clusters.size();
		
		Set<Long> toDelete = new HashSet<>();
		
		for (int r : indexes) {
			for (int i = 0; i < r; i++) {
				long index = _k(i, r);
				if (index >= 0 && index < dist.getSize()) {
					toDelete.add(index);
				}
			}

			for (int j = r + 1; j < r + 1 + (n - r - 1); j++) {
				long index = _k(r, j);
				if (index >= 0 && index < dist.getSize()) {
					toDelete.add(index);
				}
			}
			
		}
			
		List<Long> toDeleteIndexes = new ArrayList<>();
		toDeleteIndexes.addAll(toDelete);
			
		Collections.sort(toDeleteIndexes);
		
		long tot = ((long)n*(n-1))/2;
		int cntDeleted = 0;
		
		for (long it = 0; it < tot; it++) {
			if (cntDeleted == 0 && it != toDeleteIndexes.get(cntDeleted)) { continue; }
			if (cntDeleted < toDeleteIndexes.size() && it == toDeleteIndexes.get(cntDeleted))
				cntDeleted += 1;
			
			while (cntDeleted < toDeleteIndexes.size() && it + cntDeleted == toDeleteIndexes.get(cntDeleted))
				cntDeleted += 1;
			
			if (it + cntDeleted < tot)
				dist.set(it,  dist.get(it + cntDeleted));
			else break;
		}
		
		for (int i = 0; i < indexes.size(); i++) {
			int adjustedIndex = indexes.get(i) - i;
			clusters.remove(adjustedIndex);
		}
	}
	
	void insert(Cluster cluster) {
		clusters.add(0, cluster);
		int n = clusters.size();
		
		int i = 0;
		for (int j = i + 1; j < n; j++) {
			long k = _k(i, j);
			dist.set(k,  clusters.get(i).distance(clusters.get(j), d));
		}
	}
	
	List<MinDistancePair> findMinDistancePairs() {
		return FindMinDistancePairTask.findMinDistancePairs(this);
	}
	
	List<MinDistancePair> findNearestNeighbours(int clusterID) {
		return FindNearestNeighbourTask.findMinDistancePairs(this, clusterID);
	}
	
	int size() {
		return clusters.size();
	}
	
	Cluster getCluster(int i) {
		return clusters.get(i);
	}
	
	public void resize() {
		long n = this.clusters.size();
		long newSize = (n * (n - 1)) / 2;
		dist.resize(newSize);
	}
}
