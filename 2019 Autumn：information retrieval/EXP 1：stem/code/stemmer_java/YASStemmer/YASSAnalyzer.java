package YASStemmer;

import java.io.IOException;
import java.util.List;

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

import YASS.DistanceMeasure;
import stemmer.TestAnalyzer;
import stemmer.TestStemFilter;

public class YASSAnalyzer extends StopwordAnalyzerBase {
	public final static String DEFAULT_STOPWORD_FILE = "stopwords.txt";
	private final CharArraySet stemExclusionSet;
	private static List<String> lexicon;
	private DistanceMeasure d;
	private float threshold;
	
	public YASSAnalyzer(List<String> lexicon, DistanceMeasure d, float threshold) {
		this(CharArraySet.EMPTY_SET, lexicon, d, threshold);
	}
	
	public YASSAnalyzer(CharArraySet stopwords, List<String> lexicon, DistanceMeasure d, float threshold) {
		this(stopwords, CharArraySet.EMPTY_SET, lexicon, d, threshold);
	}
	
	public YASSAnalyzer(CharArraySet stopwords, CharArraySet stemExclusionTable,
			List<String> lexicon, DistanceMeasure d, float threshold) {
		super(stopwords);
		this.stemExclusionSet = CharArraySet.unmodifiableSet(CharArraySet.copy(stemExclusionTable));
		this.lexicon = lexicon;
		this.d = d;
		this.threshold = threshold;
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
		result = new YASSFilter(result, lexicon, d, threshold);
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
