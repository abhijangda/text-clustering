package com.cendrillon.clustering;

/**
 * Interface for encoders which encode documents into feature vectors.
 */
public interface Encoder {
	/** Encode all of the documents within the provided DocumentList . */
	public void encode(DocumentList documentList);
}
