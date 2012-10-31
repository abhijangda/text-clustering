package com.cendrillon.clustering;

/**
 * Implementation of Encoder which uses Term Frequency - Inverse Document Frequency (TF-IDF)
 * encoding
 */
public class TfIdfEncoder implements Encoder {
	private final int numFeatures;
	private Vector inverseDocumentFrequency;

	/**
	 * Construct a term frequency - inverse document frequency encoder. The encoder encodes documents
	 * into vectors with the specified number of features.
	 */
	public TfIdfEncoder(int numFeatures) {
		this.numFeatures = numFeatures;
	}

	/**
	 * Calculate word histogram for document and store in histogram field.
	 */
	private void calcHistogram(Document document) {
		// Calculate word histogram for document
		String[] words = document.getContents().split("[^\\w]+");
		Vector histogram = new Vector(numFeatures);
		for (int i = 0; i < words.length; i++) {
			int hashCode = hashWord(words[i]);
			histogram.increment(hashCode);
		}
		document.setHistogram(histogram);
	}

	/**
	 * Calculate word histogram for all documents in a DocumentList.
	 */
	private void calcHistogram(DocumentList documentList) {
		for (Document document : documentList) {
			calcHistogram(document);
		}
	}

	/**
	 * Calculate inverse document frequency for DocumentList. Assumes word histograms for constituent
	 * documents have already been calculated.
	 */
	private void calcInverseDocumentFrequency(DocumentList documentList) {
		inverseDocumentFrequency = new Vector(numFeatures);
		for (Document document : documentList) {
			for (int i = 0; i < numFeatures; i++) {
				if (document.getHistogram().get(i) > 0) {
					inverseDocumentFrequency.increment(i);
				}
			}
		}
		inverseDocumentFrequency.invert();
		inverseDocumentFrequency.multiply(documentList.size());
		inverseDocumentFrequency.log();
	}

	/**
	 * Encode document using Term Frequency - Inverse Document Frequency.
	 */
	private void encode(Document document) {
		// Normalize word histogram by maximum word frequency
		Vector vector = new Vector(document.getHistogram());
		// Allow histogram to be deallocated as it is no longer needed
		document.setHistogram(null);
		vector.divide(vector.max());
		// Normalize by inverseDocumentFrequency
		vector.multiply(inverseDocumentFrequency);
		// Store feature vecotr in document
		document.setVector(vector);
		// Precalculate norm for use in distance calculations
		document.setNorm(vector.norm());

	}

	/**
	 * Encode all documents within a DocumentList.
	 */
	@Override
	public void encode(DocumentList documentList) {
		calcHistogram(documentList);
		calcInverseDocumentFrequency(documentList);
		for (Document document : documentList) {
			encode(document);
		}
	}

	/**
	 * Hash word into integer between 0 and numFeatures - 1. Used to form document feature vector.
	 */
	private int hashWord(String word) {
		return Math.abs(word.hashCode()) % numFeatures;
	}
}
