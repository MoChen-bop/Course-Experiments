package stemmer;

import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.miscellaneous.SetKeywordMarkerFilter;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.KeywordAttribute;
import org.apache.lucene.util.*;

import java.io.IOException;


public final class TestStemFilter extends TokenFilter {
	private final TestStemmer stemmer = new TestStemmer();
	private final CharTermAttribute termAtt = addAttribute(CharTermAttribute.class);
	private final KeywordAttribute keywordAttr = addAttribute(KeywordAttribute.class);
	
	public TestStemFilter(TokenStream input) {
		super(input);
	}
	
	@Override
	public boolean incrementToken() throws IOException {
		if (input.incrementToken()) {
			final int newlen = stemmer.stem(termAtt.buffer(), termAtt.length());
			termAtt.setLength(newlen);
			return true;
		} else {
			return false;
		}
	}

}
