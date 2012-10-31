package com.cendrillon.clustering;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.StringTokenizer;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * A class for storing a collection of documents to be clustered.
 */
public class DocumentList implements Iterable<Document> {
	List<Document> documents = new ArrayList<Document>();

	/**
	 * Parse input string into document ID, document title and contents then encode into feature
	 * vector using encoder.
	 */
	public DocumentList(String input) {
		StringTokenizer st = new StringTokenizer(input, "{");
		int numDocuments = st.countTokens() - 1;
		String record = st.nextToken(); // skip empty split to left of {
		Pattern pattern = Pattern.compile("\"content\": \"(.*)\", \"id\": (.*), \"title\": \"(.*)\"");
		for (int i = 0; i < numDocuments; i++) {
			record = st.nextToken();
			Matcher matcher = pattern.matcher(record);
			if (matcher.find()) {
				String contents = matcher.group(1);
				long documentID = Long.parseLong(matcher.group(2));
				documents.add(new Document(documentID, contents));
			}
		}
	}

	/** Mark all documents as not being allocated to a cluster. */
	public void clearIsAllocated() {
		for (Document document : documents) {
			document.clearIsAllocated();
		}
	}

	public Document get(int index) {
		return documents.get(index);
	}

	@Override
	public Iterator<Document> iterator() {
		return documents.iterator();
	}

	public int size() {
		return documents.size();
	}
}
