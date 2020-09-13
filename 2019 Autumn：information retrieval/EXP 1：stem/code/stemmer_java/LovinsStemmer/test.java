package LovinsStemmer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileVisitResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.SimpleFileVisitor;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.Date;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.LongPoint;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig.OpenMode;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

import stemmer.TestAnalyzer;

public class test {
//	public static void main(String[] args) {
//	
//	String indexPath = "index";
//	String docsPath = "G:\\dataset\\corpus\\blogs";
//	boolean create = true;
//	
//	final Path docDir = Paths.get(docsPath);
//	
//	Date start = new Date();
//	try {
//		System.out.println("Indexing to directory '" + indexPath + "'...");
//		
//		Directory dir = FSDirectory.open(Paths.get(indexPath));
//		Analyzer analyzer = new LovinsStemAnalyzer();
//		IndexWriterConfig iwc = new IndexWriterConfig(analyzer);
//		
//		if (create) {
//			iwc.setOpenMode(OpenMode.CREATE);
//		} else {
//			iwc.setOpenMode(OpenMode.CREATE_OR_APPEND);
//		}
//		
//		IndexWriter writer = new IndexWriter(dir, iwc);
//		indexDocs(writer, docDir);
//
//		writer.close();
//		
//		Date end = new Date();
//		System.out.println(end.getTime() - start.getTime() + " total milliseconds");
//	} catch (IOException e) {
//		System.out.println(" caught a " + e.getClass() + 
//				"\n with mesage: " + e.getMessage());
//	}
//}
//
//static void indexDocs(final IndexWriter writer, Path path) throws IOException {
//	if (Files.isDirectory(path)) {
//		Files.walkFileTree(path, new SimpleFileVisitor<Path>() {
//			@Override
//			public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) throws IOException {
//				try {
//					indexDoc(writer, file, attrs.lastModifiedTime().toMillis());
//				} catch (IOException ignore) {
//					
//				}
//				return FileVisitResult.CONTINUE;
//			}
//		});
//	} else {
//		indexDoc(writer, path, Files.getLastModifiedTime(path).toMillis());
//	}
//}
//
//static void indexDoc(IndexWriter writer, Path file, long lastModified) throws IOException {
//	try (InputStream stream = Files.newInputStream(file)) {
//		Document doc = new Document();
//		Field pathField = new StringField("path", file.toString(), Field.Store.YES);
//		doc.add(pathField);
//		
//		doc.add(new LongPoint("modified", lastModified));
//		
//		doc.add(new TextField("contents", new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))));
//		
//		if (writer.getConfig().getOpenMode() == OpenMode.CREATE) {
//			System.out.println("adding " + file);
//			writer.addDocument(doc);
//		} else {
//			System.out.println("updating " + file);
//			writer.softUpdateDocument(new Term("path", file.toString()), doc);
//		}
//	}
//}
	
	public static void main(String[] args) throws Exception {
		String index = "index";
		String field = "contents";
		String queries = null;
		int repeat = 0;
		boolean raw = true;
		String queryString = null;
		int hitsPerPage = 10;

		IndexReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(index)));
		IndexSearcher searcher = new IndexSearcher(reader);
		Analyzer analyzer = new LovinsStemAnalyzer();

		BufferedReader in = null;

		if (queries != null) {
			in = Files.newBufferedReader(Paths.get(queries), StandardCharsets.UTF_8);
		} else {
			in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
		}

		QueryParser parser = new QueryParser(field, analyzer);

		while (true) {
			if (queries == null && queryString == null) {
				System.out.println("Enter query: ");
			}

			String line = queryString != null ? queryString : in.readLine();

			if (line == null || line.length() == -1) {
				break;
			}

			line = line.trim();
			if (line.length() == 0) {
				break;
			}

			Query query = parser.parse(line);
			System.out.println("Searching for: " + query.toString(field));

			if (repeat > 0) {
				Date start = new Date();
				for (int i = 0; i < repeat; i++) {
					searcher.search(query, 100);
				}
				Date end = new Date();
				System.out.println("Time: " + (end.getTime() - start.getTime()) + "ms");
			}

			doPagingSearch(in, searcher, query, hitsPerPage, raw, queries == null && queryString == null);

			if (queryString != null) {
				break;
			}
		}
		reader.close();
	}

	public static void doPagingSearch(BufferedReader in, IndexSearcher searcher, Query query, int hitsPerPage,
			boolean raw, boolean interactive) throws IOException {

		// Collect enough docs to show 5 pages
		TopDocs results = searcher.search(query, 5 * hitsPerPage);
		ScoreDoc[] hits = results.scoreDocs;

		int numTotalHits = Math.toIntExact(results.totalHits.value);
		System.out.println(numTotalHits + " total matching documents");

		int start = 0;
		int end = Math.min(numTotalHits, hitsPerPage);

		while (true) {
			if (end > hits.length) {
				System.out.println("Only results 1 - " + hits.length + " of " + numTotalHits
						+ " total matching documents collected.");
				System.out.println("Collect more (y/n) ?");
				String line = in.readLine();
				if (line.length() == 0 || line.charAt(0) == 'n') {
					break;
				}

				hits = searcher.search(query, numTotalHits).scoreDocs;
			}

			end = Math.min(hits.length, start + hitsPerPage);

			for (int i = start; i < end; i++) {
				if (raw) { // output raw format
					System.out.println("doc=" + hits[i].doc + " score=" + hits[i].score);
					continue;
				}

				Document doc = searcher.doc(hits[i].doc);
				String path = doc.get("path");
				if (path != null) {
					System.out.println((i + 1) + ". " + path);
					String title = doc.get("title");
					if (title != null) {
						System.out.println("   Title: " + doc.get("title"));
					}
				} else {
					System.out.println((i + 1) + ". " + "No path for this document");
				}

			}

			if (!interactive || end == 0) {
				break;
			}

			if (numTotalHits >= end) {
				boolean quit = false;
				while (true) {
					System.out.print("Press ");
					if (start - hitsPerPage >= 0) {
						System.out.print("(p)revious page, ");
					}
					if (start + hitsPerPage < numTotalHits) {
						System.out.print("(n)ext page, ");
					}
					System.out.println("(q)uit or enter number to jump to a page.");

					String line = in.readLine();
					if (line.length() == 0 || line.charAt(0) == 'q') {
						quit = true;
						break;
					}
					if (line.charAt(0) == 'p') {
						start = Math.max(0, start - hitsPerPage);
						break;
					} else if (line.charAt(0) == 'n') {
						if (start + hitsPerPage < numTotalHits) {
							start += hitsPerPage;
						}
						break;
					} else {
						int page = Integer.parseInt(line);
						if ((page - 1) * hitsPerPage < numTotalHits) {
							start = (page - 1) * hitsPerPage;
							break;
						} else {
							System.out.println("No such page");
						}
					}
				}
				if (quit)
					break;
				end = Math.min(numTotalHits, start + hitsPerPage);
			}
		}
	}
}