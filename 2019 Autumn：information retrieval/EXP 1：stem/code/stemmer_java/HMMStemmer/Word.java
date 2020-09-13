package HMMStemmer;

public class Word {
	private char[] buffer;
	private int[] code;
	private int maxLength;
	private int length;
	public static final int BEGINTAG = 26;
	public static final int ENDTAG = 27;
	
	public Word() {
		maxLength = 50;
		buffer = new char[maxLength];
		code = new int[maxLength];
		length = 0;
	}
	
	public Word(int maxLen) {
		maxLength = maxLen;
		buffer = new char[maxLength];
		code = new int[maxLength];
		length = 0;
	}
	
	public void add(char c) {
		buffer[length++] = c;
	}
	
	public void add(char[] chars, int l) {
		for (int i = 0; i < l; i++)
			buffer[length + i] = chars[i];
		length += l;
	}
	
	public int Length() {
		return length;
	}
	
	public char getCharAt(int index) {
		return buffer[index];
	}
	
	public int getCodeAt(int index) {
		return code[index];
	}
	
	public char[] getChar() {
		char[] word = new char[length];
		for (int i = 0; i < length; i++)
			word[i] = buffer[i];
		return word;
	}
	
	public int[] getCode() {
		encode();
		int[] c = new int[length + 2];
		for (int i = 0; i <= length + 1; i++)
			c[i] = code[i];
		return c;
	}
	
	public void encode() {
		code[0] = BEGINTAG;
		for (int i = 0; i < length; i++) {
			code[i + 1] = (int)(buffer[i] - 97);
		}
		code[length + 1] = ENDTAG;
	}
	
	public String toString() {
		return new String(buffer, 0, length);
	}
	
}
