package com.cendrillon.clustering;

/**
 * Class for calculating cosine distance between two vectors.
 */
public class CosineDistance extends DistanceMetric {
	@Override
	protected double calcDistance(Vector vector1, Vector vector2, double norm1, double norm2) {
		return 1 - vector1.innerProduct(vector2) / norm1 / norm2;
	}
}
