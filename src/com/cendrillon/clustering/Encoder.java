package com.cendrillon.clustering;

/**
 * Interface for encoders which encode documents into feature vectors.
 */
public interface Encoder {
	public void encode(DocumentList documentList);
}
