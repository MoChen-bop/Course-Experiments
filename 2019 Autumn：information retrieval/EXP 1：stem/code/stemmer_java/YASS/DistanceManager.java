package YASS;

import java.util.ArrayList;
import java.util.List;

public class DistanceManager {
	
	private static float INFINITY = Float.POSITIVE_INFINITY;
	
	private static int p(String x, String y, int i) {
		int minLen = Math.min(x.length(), y.length());
		if (i >= minLen) {
			return 1;
		}
		if (x.charAt(i) == y.charAt(i)) {
			return 0;
		}
		return 1;
	}
	
	private static int firstMismatch(String x, String y) {
		int minLen = Math.min(x.length(), y.length());
		for (int i = 0; i < minLen; i++) {
			if (x.charAt(i) != y.charAt(i)) {
				return i;
			}
		}
		return minLen;
	}
	
	private static List<String> getNGrams(String word, int n) {
		List<String> grams = new ArrayList<>();
		if (word.length() <= n) {
			grams.add(word);
			return grams;
		}
		
		for (int begin = -n + 1; begin < word.length(); begin++) {
			String gram = getGram(word, begin, n);
			grams.add(gram);
		}
		
		return grams;
	}

	private static String getGram(String word, int begin, int n) {
		int left = begin;
		int right = begin + n;
		String stem = "";
		while (left < 0) {
			stem += "-";
			left++;
		}
		
		if (right < word.length()) {
			stem += word.substring(left, right);
		} else {
			stem += word.substring(left);
			left += word.length() - left;
			while (left < right) {
				stem += "-";
				left++;
			}
		}
		return stem;
	}
	
	private static float calculateJaccardIndex(List<String> grams1, List<String> grams2) {
		int count = 0;
		for (String gram : grams1) {
			if (grams2.contains(gram)) count++;
		}
		return count / (grams1.size() + grams2.size() - count);
	}
	
	public static DistanceMeasure d1() {
		return new DistanceMeasure() {
			@Override
			public float calculate(String w1, String w2) {
				int maxLen = Math.max(w1.length(), w2.length());
				float d = 0;
				for (int i = 0; i < maxLen; i++) {
					d += p(w1, w2, i) / Math.pow(2,  i);
				}
				return d;
			}
			@Override
			public String getName() {
				return "d1";
			}
		};
	}
	
	public static DistanceMeasure d2() {
		return new DistanceMeasure() {
			@Override
			public float calculate(String w1, String w2) {
				int maxLen = Math.max(w1.length(), w2.length());
				float d = 0;
				int m = firstMismatch(w1, w2);
				
				if (m == 0) { return INFINITY; }
				for (int i = m; i < maxLen; i++) {
					d += 1 / Math.pow(2,  i - m);
				}
				
				return d / (float)m;
			}
			@Override
			public String getName() {
				return "d2";
			}
		};
	}
	
	public static DistanceMeasure d3() {
        return new DistanceMeasure() {
            @Override
            public float calculate(String w1, String w2) {
                int maxLen = Math.max(w1.length(), w2.length());
                float d = 0;
                int n = maxLen -1;
                int m = firstMismatch(w1,w2);

                if (m == 0){ return INFINITY;}
                for (int i = m; i < maxLen; i++) {
                    d += 1 / Math.pow(2,i-m);
                }

                return (d * (n-m+1)) / (float)m;
            }
            @Override
            public String getName() {
                return "d3";
            }
        };
    }

    public static DistanceMeasure d4() {
        return new DistanceMeasure() {
            @Override
            public float calculate(String w1, String w2) {
                int maxLen = Math.max(w1.length(), w2.length());
                float d = 0;
                int m = firstMismatch(w1,w2);
                int n = maxLen - 1;

                if (m == 0){ return INFINITY;}
                for (int i = m; i < maxLen; i++) {
                    d += 1 / Math.pow(2,i-m);
                }

                return (d * (n-m+1)) / (float)(n+1);
            }

            @Override
            public String getName() {
                return "d4";
            }
        };
    }
    
    public static DistanceMeasure d5(int n) {
    	return new DistanceMeasure() {
    		 @Override
             public float calculate(String w1, String w2) {
    			 List<String> grams1 = getNGrams(w1, n);
    			 List<String> grams2 = getNGrams(w2, n);
    			 
    			 return calculateJaccardIndex(grams1, grams2);
             }

             @Override
             public String getName() {
                 return "d5-" + n;
             }
    	};
    }
}
