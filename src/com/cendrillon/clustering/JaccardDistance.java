package com.cendrillon.clustering;

/**
 * Class for caculating Jaccard distance between vectors.
 */
public class JaccardDistance extends DistanceMetric {
	@Override
	protected double calcDistance(Vector vector1, Vector vector2) {
		double innerProduct = vector1.innerProduct(vector2);
		return Math.abs(1 - innerProduct / (vector1.norm() + vector2.norm() - innerProduct));
	}
}
