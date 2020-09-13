package YASStemmer;

import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.miscellaneous.SetKeywordMarkerFilter;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.KeywordAttribute;
import org.apache.lucene.util.*;

import YASS.DistanceMeasure;
import stemmer.TestStemmer;

import java.io.IOException;
import java.util.List;

public class YASSFilter extends TokenFilter {
	private YASStemmer stemmer;
	private final CharTermAttribute termAtt = addAttribute(CharTermAttribute.class);
	private final KeywordAttribute keywordAttr = addAttribute(KeywordAttribute.class);
	
	public YASSFilter(TokenStream input, List<String> lexicon, DistanceMeasure d, float threshold) {
		super(input);
		stemmer = new YASStemmer(lexicon, d, threshold);
	}
	
	@Override
	public boolean incrementToken() throws IOException {
		if (input.incrementToken()) {
			String word = new String(termAtt.buffer(), 0, termAtt.length());
			String stem = stemmer.stem(word);
			final int newlen = stem.length();
			termAtt.setLength(newlen);
			return true;
		} else {
			return false;
		}
	}
}
