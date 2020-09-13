package clustering;

public class MinDistancePair {
	private int r;
	private int s;
	private float dist;
	
	public MinDistancePair(int r, int s, float dist) {
		this.r = r;
		this.s = s;
		this.dist = dist;
	}
	
	public int getR() {
		return r;
	}
	
	public int getS() {
		return s;
	}
	
	public float getDist() {
		return dist;
	}

	public int compareTo(MinDistancePair aPair) {
		if (this.dist < aPair.dist) {
			return -1;
		}
		if (this.dist == aPair.dist) {
			return 0;
		}

		return 1;
	}
	
	@Override
	public String toString() {
		return "MinDistencePair{" + 
				"r=" + r + ", s=" + s +
				", dist=" + dist + "}";
	}
}
