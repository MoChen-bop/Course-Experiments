package HMMStemmer;

import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.miscellaneous.SetKeywordMarkerFilter;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.KeywordAttribute;
import org.apache.lucene.util.*;

import stemmer.TestStemmer;

import java.io.IOException;

public class HMMStemFilter extends TokenFilter {
	private HMMStemmer stemmer;
	private final CharTermAttribute termAtt = addAttribute(CharTermAttribute.class);
	private final KeywordAttribute keywordAttr = addAttribute(KeywordAttribute.class);
	
	public HMMStemFilter(TokenStream input, HMMStemmer stemmer) {
		super(input);
		this.stemmer = stemmer;
	}
	
	public boolean isValid(char[] arr, int len) {
		for (int i = 0; i < len; i++) {
			if (arr[i] > (int)('z') || arr[i] < (int)('a'))
				return false;
		}
		return true;
	}
	
	@Override
	public boolean incrementToken() throws IOException {
		if (input.incrementToken()) {
			if (isValid(termAtt.buffer(), termAtt.length())) {
				final int newlen = stemmer.stem(termAtt.buffer(), termAtt.length());
				termAtt.setLength(newlen);
			}
			return true;
		} else {
			return false;
		}
	}
}
