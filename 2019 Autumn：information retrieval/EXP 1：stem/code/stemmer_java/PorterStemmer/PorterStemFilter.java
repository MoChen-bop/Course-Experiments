package PorterStemmer;

import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.miscellaneous.SetKeywordMarkerFilter;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.KeywordAttribute;
import org.apache.lucene.util.*;

import stemmer.TestStemmer;

import java.io.IOException;

public class PorterStemFilter extends TokenFilter {
	private final PorterStemmer stemmer = new PorterStemmer();
	private final CharTermAttribute termAtt = addAttribute(CharTermAttribute.class);
	private final KeywordAttribute keywordAttr = addAttribute(KeywordAttribute.class);
	
	public PorterStemFilter(TokenStream input) {
		super(input);
	}
	
	@Override
	public boolean incrementToken() throws IOException {
		if (input.incrementToken()) {
			stemmer.add(termAtt.buffer(), termAtt.length());
			stemmer.stem();
			final int newlen = stemmer.getResultLength();
			termAtt.setLength(newlen);
			return true;
		} else {
			return false;
		}
	}
}
