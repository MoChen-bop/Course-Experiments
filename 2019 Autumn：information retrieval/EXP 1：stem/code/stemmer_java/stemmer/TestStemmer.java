package stemmer;

public class TestStemmer {
	
	public int stem(char s[], int len) {
		len = swapFirstCharacter(s, len);
		len = removeEndCharacter(s, len);
		return len;
	}
	
	private int swapFirstCharacter(char[] s, int len) {
		if (len > 1) {
			s[0] = s[len - 1];
		}
		
		return len;
	}
	
	private int removeEndCharacter(char[] s, int len) {
		if (len > 0) {
			len = len - 1;
		}
		
		return len;
	}
	
}
