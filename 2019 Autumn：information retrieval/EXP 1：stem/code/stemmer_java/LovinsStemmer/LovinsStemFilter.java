package LovinsStemmer;

import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.miscellaneous.SetKeywordMarkerFilter;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.KeywordAttribute;
import org.apache.lucene.util.*;

import stemmer.TestStemmer;

import java.io.IOException;

public class LovinsStemFilter extends TokenFilter {
	private final LovinsStemmer stemmer = new LovinsStemmer();
	private final CharTermAttribute termAtt = addAttribute(CharTermAttribute.class);
	private final KeywordAttribute keywordAttr = addAttribute(KeywordAttribute.class);
	
	public LovinsStemFilter(TokenStream input) {
		super(input);
	}
	
	@Override
	public boolean incrementToken() throws IOException {
		if (input.incrementToken()) {
			String result = stemmer.stem(String.copyValueOf(termAtt.buffer(), 0, termAtt.length()));
			final int newlen = result.length();
			System.arraycopy(result.toCharArray(), 0, termAtt.buffer(), 0, newlen);
			termAtt.setLength(newlen);
			return true;
		} else {
			return false;
		}
	}
}