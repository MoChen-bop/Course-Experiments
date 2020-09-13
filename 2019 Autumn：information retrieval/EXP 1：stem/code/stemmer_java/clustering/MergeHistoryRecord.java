package clustering;

public class MergeHistoryRecord implements Comparable {
	private int c1;
	private int c2;
	private int cres;
	private float dist;
	private int cnt;
	
	public MergeHistoryRecord(int c1, int c2, int cres, float dist, int cnt) {
		this.c1 = c1;
		this.c2 = c2;
		this.cres = cres;
		this.dist = dist;
		this.cnt = cnt;
	}
	
	public int getC1() {
		return c1;
	}
	
	public int getC2() {
		return c2;
	}
	
	public float getDist() {
		return dist;
	}
	
	public int getCnt() {
		return cnt;
	}
	
	@Override
	public String toString() {
		return "MergeHistory{" + 
				"c1=" + c1 + ", c2=" + c2 +
				", cres=" + cres + ", dist=" + dist +
				", cnt=" + cnt + "}";
	}
	
	@Override
	public boolean equals(Object o) {
		if (this == o) return true;
		if (o == null || getClass() != o.getClass()) return false;
		
		MergeHistoryRecord that = (MergeHistoryRecord) o;
		
		if (c1 != that.c1) return false;
		if (c2 != that.c2) return false;
		if (cres != that.cres) return false;
		return Float.compare(this.dist, dist) == 0 && cnt == that.cnt;
	}
	
	@Override
	public int compareTo(Object o) {
		MergeHistoryRecord record = (MergeHistoryRecord) o;
		
		if (this.dist == record.dist) {
			return 0;
		}
		if (this.dist < record.dist) {
			return -1;
		} else {
			return 1;
		}
	}

	public void setC1(int c1) {
		this.c1 = c1;
	}

	public void setC2(int c2) {
		this.c2 = c2;
	}

	public void setCres(int cres) {
		this.cres = cres;
	}

	public void setDist(float dist) {
		this.dist = dist;
	}

	public void setCnt(int cnt) {
		this.cnt = cnt;
	}
	
	
}
