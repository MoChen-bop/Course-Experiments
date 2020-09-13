package PorterStemmer;

import java.io.IOException;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.StopwordAnalyzerBase;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.Analyzer.TokenStreamComponents;
import org.apache.lucene.analysis.core.DecimalDigitFilter;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.core.StopFilter;
import org.apache.lucene.analysis.miscellaneous.SetKeywordMarkerFilter;
import org.apache.lucene.analysis.standard.StandardTokenizer;

import stemmer.TestAnalyzer;
import stemmer.TestStemFilter;

public class PorterStemAnalyzer extends StopwordAnalyzerBase {
	public final static String DEFAULT_STOPWORD_FILE = "stopwords.txt";
	private final CharArraySet stemExclusionSet;
	
	public PorterStemAnalyzer() {
		this(CharArraySet.EMPTY_SET);
	}
	
	public PorterStemAnalyzer(CharArraySet stopwords) {
		this(stopwords, CharArraySet.EMPTY_SET);
	}
	
	public PorterStemAnalyzer(CharArraySet stopwords, CharArraySet stemExclusionTable) {
		super(stopwords);
		this.stemExclusionSet = CharArraySet.unmodifiableSet(CharArraySet.copy(stemExclusionTable));
	}
	
	@Override
	protected TokenStreamComponents createComponents(String fieldName) {
		final Tokenizer source = new StandardTokenizer();
		TokenStream result = new LowerCaseFilter(source);
		result = new DecimalDigitFilter(result);
		result = new StopFilter(result, stopwords);
		if (!stemExclusionSet.isEmpty()) {
			result = new SetKeywordMarkerFilter(result, stemExclusionSet);
		}
		result = new PorterStemFilter(result);
		return new TokenStreamComponents(source, result);
	}
	
	private static class DefaultSetHolder {
		static final CharArraySet DEFAULT_STOP_SET;
		
		static {
			try {
				DEFAULT_STOP_SET = loadStopwordSet(false, TestAnalyzer.class,
						DEFAULT_STOPWORD_FILE, "#");
			} catch (IOException ex) {
				throw new RuntimeException("Unable to load default stopword set");
			}
		}
	}
}
